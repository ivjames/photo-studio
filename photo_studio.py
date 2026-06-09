#!/usr/bin/env python3
"""
photo_studio.py — local web app for splitting & tagging scanned photos.

Wraps the two-stage pipeline (scan_splitter.py + photo_tagger.py) in a browser
UI: load a scan or folder, review auto-detected crops, fix rotations, flag
blanks/duplicates, auto-tag with Claude, then export corrected photos + a
manifest. Runs entirely on your machine.

Requires scan_splitter.py and photo_tagger.py in the same folder.

Run:
    pip install flask opencv-python numpy
    python photo_studio.py            # opens http://127.0.0.1:5000
    python photo_studio.py --port 8080 --no-browser

Tagging (optional) uses the Anthropic API over plain HTTPS — no SDK needed.
Provide a key in the UI or via:  export ANTHROPIC_API_KEY=sk-...
"""

import argparse
import base64
import csv
import hmac
import io
import json
import os
import tempfile
import threading
import zipfile
import urllib.request
import urllib.error
import uuid
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, Response, after_this_request

# Reuse the validated CV logic from the two pipeline scripts.
from scan_splitter import estimate_background, build_foreground_mask, crop_rotated
from photo_tagger import (
    blankness, dhash, group_duplicates, face_orientation_hint,
    encode_image, parse_json, VISION_PROMPT, apply_rotation, IMAGE_EXTS,
)

app = Flask(__name__)
# Stable work dir so crops + session survive restarts (override with PHOTOSTUDIO_HOME).
WORK = Path(os.environ.get("PHOTOSTUDIO_HOME", Path.home() / ".photostudio"))
WORK.mkdir(parents=True, exist_ok=True)
SESSION_FILE = WORK / "session.json"
STATE = {"scans": [], "photos": [], "rev": 0}  # shared workspace, persisted to disk
SAVE_LOCK = threading.Lock()
DEFAULT_MODEL = "claude-sonnet-4-5"   # confirm current id at docs.claude.com

# --- deployment config (env vars; sane local defaults) ---------------------
TAGGER = os.environ.get("PHOTOSTUDIO_TAGGER", "anthropic").lower()  # "anthropic" | "ollama"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4")  # must be a vision model
APP_USER = os.environ.get("PHOTOSTUDIO_USER", "admin")
APP_PASSWORD = os.environ.get("PHOTOSTUDIO_PASSWORD", "")  # single-account convenience


def _load_users():
    """Accounts share one workspace. From PHOTOSTUDIO_USERS='alice:pw1,bob:pw2'
    plus the single PHOTOSTUDIO_USER/PASSWORD pair. Empty = no auth (local use)."""
    users = {}
    for pair in os.environ.get("PHOTOSTUDIO_USERS", "").split(","):
        if ":" in pair:
            u, pw = pair.split(":", 1)
            users[u.strip()] = pw.strip()
    if APP_PASSWORD:
        users[APP_USER] = APP_PASSWORD
    return users


USERS = _load_users()


def current_user():
    a = request.authorization
    return a.username if a else "local"


@app.before_request
def _require_auth():
    """When any accounts are configured (i.e. deployed), require HTTP Basic auth."""
    if not USERS:
        return  # local use: no auth
    if request.path == "/health":
        return  # health check must always be reachable
    a = request.authorization
    ok = a and a.username in USERS and hmac.compare_digest(a.password or "", USERS[a.username])
    if not ok:
        return Response("Authentication required.", 401,
                        {"WWW-Authenticate": 'Basic realm="Photo Studio"'})


def save_session():
    """Persist metadata + bump a revision counter so clients can poll for changes.
    Atomic write under a small lock so concurrent users can't corrupt the file."""
    STATE["rev"] = STATE.get("rev", 0) + 1
    try:
        data = json.dumps(STATE, default=str)
        with SAVE_LOCK:
            tmp = SESSION_FILE.with_suffix(".tmp")
            tmp.write_text(data)
            os.replace(tmp, SESSION_FILE)
    except Exception as e:  # noqa: BLE001
        print("  ! could not save session:", e)


def load_session():
    """Reload a previous session, dropping entries whose image files are gone."""
    if not SESSION_FILE.exists():
        return
    try:
        data = json.loads(SESSION_FILE.read_text())
        STATE["scans"] = [s for s in data.get("scans", []) if Path(s["path"]).exists()]
        STATE["photos"] = [p for p in data.get("photos", []) if Path(p["path"]).exists()]
        print(f"  restored session: {len(STATE['scans'])} scan(s), "
              f"{len(STATE['photos'])} photo(s)")
    except Exception as e:  # noqa: BLE001
        print("  ! could not load session:", e)


# --------------------------------------------------------------------------- #
# Detection — returns crops plus geometry so the UI can show overlays
# --------------------------------------------------------------------------- #
def detect_photos(img, sensitivity, min_area_frac, max_area_frac, pad, deskew):
    bg = estimate_background(img)
    mask = build_foreground_mask(img, bg, sensitivity)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = img.shape[:2]
    total = H * W
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_frac * total or area > max_area_frac * total:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        if deskew:
            rect = cv2.minAreaRect(c)
            (cx, cy), (rw, rh), ang = rect
            crop = crop_rotated(img, ((cx, cy), (rw + 2 * pad, rh + 2 * pad), ang),
                                bw / bh if bh else 1.0)
        else:
            x0, y0 = max(0, bx - pad), max(0, by - pad)
            x1, y1 = min(W, bx + bw + pad), min(H, by + bh + pad)
            crop = img[y0:y1, x0:x1]
        if crop is None or crop.size == 0:
            continue
        out.append((by, bx, crop, [int(bx), int(by), int(bw), int(bh)]))
    # reading order: top band, then left-to-right
    out.sort(key=lambda t: (round(t[0] / (H * 0.1)), t[1]))
    return out


def register_scan(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    sid = uuid.uuid4().hex[:8]
    dst = WORK / f"scan_{sid}.jpg"
    cv2.imwrite(str(dst), img)
    STATE["scans"].append({"id": sid, "name": Path(path).name,
                           "path": str(dst), "w": img.shape[1], "h": img.shape[0]})
    return sid


def run_detection(params):
    STATE["photos"] = []
    grays = []
    for scan in STATE["scans"]:
        img = cv2.imread(scan["path"])
        found = detect_photos(
            img,
            sensitivity=params["sensitivity"],
            min_area_frac=params["min_area_frac"],
            max_area_frac=0.95,
            pad=params["pad"],
            deskew=params["deskew"],
        )
        for i, (_, _, crop, bbox) in enumerate(found, 1):
            pid = uuid.uuid4().hex[:8]
            cpath = WORK / f"crop_{pid}.jpg"
            cv2.imwrite(str(cpath), crop)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            is_blank, std, edge = blankness(gray)
            grays.append((pid, gray))
            bx, by, bw, bh = bbox
            STATE["photos"].append({
                "id": pid, "scan_id": scan["id"], "scan_name": scan["name"],
                "index": i, "path": str(cpath), "bbox": bbox,
                "origin_quad": [[float(bx), float(by)], [float(bx + bw), float(by)],
                                [float(bx + bw), float(by + bh)], [float(bx), float(by + bh)]],
                "w": crop.shape[1], "h": crop.shape[0],
                "blank": bool(is_blank), "std": round(std, 1),
                "rotation_cw": 0, "dup_group": None, "is_dup_copy": False,
                "face_hint_cw": None,
                "status": "skip" if is_blank else "keep",
                "tags": [], "description": "", "scene_type": "",
                "people_count": None, "decade": "", "decade_confidence": "",
                "added_by": current_user(),
            })
    # duplicates across everything
    if grays:
        hashes = [dhash(g) for _, g in grays]
        groups = group_duplicates(hashes, params["dup_dist"])
        seen = set()
        gid_by_pid = {}
        for (pid, _), g in zip(grays, groups):
            gid_by_pid[pid] = g
        for p in STATE["photos"]:
            g = gid_by_pid[p["id"]]
            p["dup_group"] = int(g)
            p["is_dup_copy"] = g in seen
            seen.add(g)
            if p["is_dup_copy"] and p["status"] == "keep":
                p["status"] = "skip"
        # offline orientation hint for keepers
        gray_by_pid = dict(grays)
        for p in STATE["photos"]:
            if not p["blank"] and not p["is_dup_copy"]:
                # Stored for reference/Claude; not auto-applied (unreliable on real photos).
                p["face_hint_cw"] = face_orientation_hint(gray_by_pid[p["id"]])
    return STATE["photos"]


# --------------------------------------------------------------------------- #
# Claude tagging over plain HTTPS (no SDK dependency)
# --------------------------------------------------------------------------- #
def claude_tag(bgr, face_hint, api_key, model, max_dim=1024):
    hint = (f"A face-detector suggests upright = {face_hint} deg clockwise; "
            f"confirm or override." if face_hint is not None
            else "No faces detected automatically; use horizon/text cues.")
    b64 = encode_image(bgr, max_dim)
    body = json.dumps({
        "model": model, "max_tokens": 600,
        "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": VISION_PROMPT.format(hint=hint)},
        ]}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read())
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text")
        return parse_json(text)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def ollama_tag(bgr, face_hint, model=None, max_dim=1024):
    """Tag via a (remote) Ollama vision model, e.g. gemma4. Sends base64 image
    in the /api/chat 'images' array; asks for JSON output."""
    hint = (f"A face-detector suggests upright = {face_hint} deg clockwise; "
            f"confirm or override." if face_hint is not None
            else "No faces detected automatically; use horizon/text cues.")
    b64 = encode_image(bgr, max_dim)
    body = json.dumps({
        "model": model or OLLAMA_MODEL,
        "messages": [{"role": "user", "content": VISION_PROMPT.format(hint=hint),
                      "images": [b64]}],
        "stream": False, "format": "json", "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:  # home CPU can be slow
            data = json.loads(r.read())
        return parse_json(data.get("message", {}).get("content", ""))
    except urllib.error.URLError as e:
        return {"error": f"Ollama unreachable at {OLLAMA_URL}: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
# --------------------------------------------------------------------------- #
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/api/state")
def state():
    return jsonify({"scans": STATE["scans"], "photos": STATE["photos"],
                    "rev": STATE.get("rev", 0)})


@app.route("/api/rev")
def rev():
    return jsonify({"rev": STATE.get("rev", 0)})


@app.route("/api/config")
def config():
    return jsonify({"user": current_user(), "multiuser": bool(USERS)})


@app.route("/api/clear", methods=["POST"])
def clear():
    STATE["scans"] = []
    STATE["photos"] = []
    for pat in ("crop_*.jpg", "scan_*.jpg", "upload_*"):
        for f in WORK.glob(pat):
            try:
                f.unlink()
            except OSError:
                pass
    save_session()
    return jsonify({"ok": True})


@app.route("/api/load_folder", methods=["POST"])
def load_folder():
    folder = Path(request.json.get("path", "")).expanduser()
    if not folder.is_dir():
        return jsonify({"error": f"Not a folder: {folder}"}), 400
    STATE["scans"] = []
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    for f in files:
        register_scan(f)
    save_session()
    return jsonify({"scans": STATE["scans"]})


@app.route("/api/upload", methods=["POST"])
def upload():
    STATE["scans"] = []
    for f in request.files.getlist("files"):
        tmp = WORK / f"upload_{uuid.uuid4().hex[:8]}_{f.filename}"
        f.save(str(tmp))
        register_scan(tmp)
    save_session()
    return jsonify({"scans": STATE["scans"]})


@app.route("/api/detect", methods=["POST"])
def detect():
    j = request.json or {}
    params = {
        "sensitivity": float(j.get("sensitivity", 1.0)),
        "min_area_frac": float(j.get("min_area_frac", 0.01)),
        "pad": int(j.get("pad", 6)),
        "deskew": bool(j.get("deskew", True)),
        "dup_dist": int(j.get("dup_dist", 10)),
    }
    photos = run_detection(params)
    save_session()
    return jsonify({"photos": photos})


def recompute_duplicates(dup_dist=10):
    photos = STATE["photos"]
    if not photos:
        return
    hashes = [dhash(cv2.cvtColor(cv2.imread(p["path"]), cv2.COLOR_BGR2GRAY)) for p in photos]
    groups = group_duplicates(hashes, dup_dist)
    seen = set()
    for p, gp in zip(photos, groups):
        p["dup_group"] = int(gp)
        p["is_dup_copy"] = gp in seen
        seen.add(gp)


def order_quad(pts):
    """Order 4 points as TL, TR, BR, BL regardless of click order."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)


def warp_quad(img, pts):
    """Perspective-warp a 4-corner photo region into an upright rectangle (deskew)."""
    q = order_quad(pts)
    tl, tr, br, bl = q
    wid = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    hei = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if wid < 5 or hei < 5:
        return None
    dst = np.array([[0, 0], [wid - 1, 0], [wid - 1, hei - 1], [0, hei - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (wid, hei))


@app.route("/api/crop_manual", methods=["POST"])
def crop_manual():
    j = request.json or {}
    sid = j.get("scan_id")
    quads = j.get("quads", [])
    scan = next((s for s in STATE["scans"] if s["id"] == sid), None)
    if not scan:
        return jsonify({"error": "scan not found"}), 404
    img = cv2.imread(scan["path"])
    base = sum(1 for p in STATE["photos"] if p["scan_id"] == sid)
    added = 0
    for k, quad in enumerate(quads, 1):
        if len(quad) != 4:
            continue
        crop = warp_quad(img, quad)
        if crop is None or crop.size == 0:
            continue
        pid = uuid.uuid4().hex[:8]
        cpath = WORK / f"crop_{pid}.jpg"
        cv2.imwrite(str(cpath), crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        is_blank, std, _ = blankness(gray)
        # Face hint is stored for reference/Claude but NOT auto-applied (unreliable
        # on faded, angled prints); the user sets final rotation in the grid.
        hint = None if is_blank else face_orientation_hint(gray)
        STATE["photos"].append({
            "id": pid, "scan_id": sid, "scan_name": scan["name"],
            "index": base + k, "path": str(cpath), "bbox": None,
            "origin_quad": [[float(x), float(y)] for x, y in quad],
            "w": crop.shape[1], "h": crop.shape[0],
            "blank": bool(is_blank), "std": round(std, 1),
            "rotation_cw": 0, "dup_group": None, "is_dup_copy": False,
            "face_hint_cw": hint, "status": "skip" if is_blank else "keep",
            "tags": [], "description": "", "scene_type": "",
            "people_count": None, "decade": "", "decade_confidence": "",
            "added_by": current_user(),
        })
        added += 1
    recompute_duplicates()
    save_session()
    return jsonify({"photos": STATE["photos"], "added": added})


@app.route("/api/delete_crop", methods=["POST"])
def delete_crop():
    """Delete a single crop: remove it from state, delete its file, and
    recompute duplicates. Its dimmed region on the parent scan clears because
    the editor derives dimming live from the remaining crops' origin_quads."""
    pid = (request.json or {}).get("id")
    p = next((x for x in STATE["photos"] if x["id"] == pid), None)
    if not p:
        return jsonify({"error": "not found"}), 404
    try:
        Path(p["path"]).unlink()
    except OSError:
        pass
    STATE["photos"] = [x for x in STATE["photos"] if x["id"] != pid]
    recompute_duplicates()
    save_session()
    return jsonify({"ok": True, "photos": STATE["photos"]})


@app.route("/api/thumb/<pid>")
def thumb(pid):
    """Downscaled image for the grid (longest edge 480px) — fast to scroll."""
    p = next((x for x in STATE["photos"] if x["id"] == pid), None)
    if not p:
        return "", 404
    img = apply_rotation(cv2.imread(p["path"]), p["rotation_cw"])
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > 320:
        s = 320 / longest
        img = cv2.resize(img, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/full/<pid>")
def full(pid):
    """Full-resolution image (rotation applied) for the lightbox view."""
    p = next((x for x in STATE["photos"] if x["id"] == pid), None)
    if not p:
        return "", 404
    img = apply_rotation(cv2.imread(p["path"]), p["rotation_cw"])
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/jpeg")


@app.route("/api/scan/<sid>")
def scan_img(sid):
    s = next((x for x in STATE["scans"] if x["id"] == sid), None)
    return send_file(s["path"], mimetype="image/jpeg") if s else ("", 404)


@app.route("/api/update", methods=["POST"])
def update():
    j = request.json
    p = next((x for x in STATE["photos"] if x["id"] == j["id"]), None)
    if not p:
        return jsonify({"error": "not found"}), 404
    for k in ("rotation_cw", "status", "description", "scene_type",
              "people_count", "decade"):
        if k in j:
            p[k] = j[k]
    if "tags" in j:
        p["tags"] = [t.strip() for t in j["tags"] if t.strip()] \
            if isinstance(j["tags"], list) else \
            [t.strip() for t in str(j["tags"]).split(",") if t.strip()]
    save_session()
    return jsonify({"photo": p})


@app.route("/api/tag", methods=["POST"])
def tag():
    j = request.json or {}
    provider = (j.get("provider") or TAGGER).lower()
    if provider == "ollama":
        model = j.get("model") or OLLAMA_MODEL
        run = lambda bgr, hint: ollama_tag(bgr, hint, model)  # noqa: E731
    else:
        key = j.get("api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            return jsonify({"error": "No API key provided."}), 400
        model = j.get("model") or DEFAULT_MODEL
        run = lambda bgr, hint: claude_tag(bgr, hint, key, model)  # noqa: E731
    ids = j.get("ids") or [p["id"] for p in STATE["photos"]
                           if p["status"] == "keep" and not p["blank"]]
    done = []
    for pid in ids:
        p = next((x for x in STATE["photos"] if x["id"] == pid), None)
        if not p:
            continue
        meta = run(cv2.imread(p["path"]), p["face_hint_cw"])
        if "error" in meta:
            p["tag_error"] = meta["error"]
            done.append({"id": pid, "error": meta["error"]})
            continue
        p["tags"] = meta.get("tags", [])
        p["description"] = meta.get("description", "")
        p["scene_type"] = meta.get("scene_type", "")
        p["people_count"] = meta.get("people_count")
        p["decade"] = meta.get("estimated_decade", "")
        p["decade_confidence"] = meta.get("decade_confidence", "")
        rot = meta.get("correct_rotation_cw", p["rotation_cw"])
        if isinstance(rot, (int, float)):
            p["rotation_cw"] = int(rot) % 360
        done.append({"id": pid, "ok": True})
    save_session()
    return jsonify({"results": done, "photos": STATE["photos"]})


@app.route("/api/export", methods=["POST"])
def export():
    out = Path((request.json or {}).get("out_dir", "")).expanduser()
    if not str(out):
        return jsonify({"error": "No output folder."}), 400
    (out / "corrected").mkdir(parents=True, exist_ok=True)
    (out / "blank").mkdir(parents=True, exist_ok=True)
    rows, n = [], 0
    for p in STATE["photos"]:
        img = cv2.imread(p["path"])
        if p["blank"] or p["status"] == "skip":
            cv2.imwrite(str(out / "blank" / f"{p['scan_name']}_{p['id']}.jpg"), img)
        else:
            corrected = apply_rotation(img, p["rotation_cw"])
            name = f"{Path(p['scan_name']).stem}_{p['index']:02d}_{p['id']}.jpg"
            cv2.imwrite(str(out / "corrected" / name), corrected)
            n += 1
        rows.append({k: p.get(k) for k in (
            "scan_name", "index", "id", "status", "blank", "dup_group",
            "is_dup_copy", "rotation_cw", "scene_type", "people_count",
            "decade", "decade_confidence", "description", "tags")})
    with open(out / "manifest.json", "w") as f:
        json.dump(rows, f, indent=2, default=str)
    keys = list(rows[0].keys()) if rows else []
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list) else r[k])
                        for k in keys})
    return jsonify({"exported": n, "total": len(rows), "out_dir": str(out)})


@app.route("/api/download", methods=["POST"])
def download():
    """Build the same export (corrected/ + blank/ + manifests) as a zip and
    stream it to the browser. No server path needed — for remote deployments
    where the user can't reach the server filesystem.

    The zip is written to a temp file (not held in memory) so peak memory stays
    near one image at a time, matching /api/export's footprint."""
    photos = STATE["photos"]
    if not photos:
        return jsonify({"error": "Nothing to download — no photos."}), 400

    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp_path = tmp.name
    tmp.close()
    rows, n = [], 0
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in photos:
                img = cv2.imread(p["path"])
                if img is None:
                    continue
                if p["blank"] or p["status"] == "skip":
                    arc = f"blank/{p['scan_name']}_{p['id']}.jpg"
                    ok, buf = cv2.imencode(".jpg", img,
                                           [cv2.IMWRITE_JPEG_QUALITY, 92])
                else:
                    corrected = apply_rotation(img, p["rotation_cw"])
                    arc = (f"corrected/{Path(p['scan_name']).stem}"
                           f"_{p['index']:02d}_{p['id']}.jpg")
                    ok, buf = cv2.imencode(".jpg", corrected,
                                           [cv2.IMWRITE_JPEG_QUALITY, 92])
                    n += 1
                if ok:
                    zf.writestr(arc, buf.tobytes())
                rows.append({k: p.get(k) for k in (
                    "scan_name", "index", "id", "status", "blank", "dup_group",
                    "is_dup_copy", "rotation_cw", "scene_type", "people_count",
                    "decade", "decade_confidence", "description", "tags")})
            zf.writestr("manifest.json",
                        json.dumps(rows, indent=2, default=str))
            keys = list(rows[0].keys()) if rows else []
            sio = io.StringIO()
            w = csv.DictWriter(sio, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (json.dumps(r[k]) if isinstance(r[k], list)
                                else r[k]) for k in keys})
            zf.writestr("manifest.csv", sio.getvalue())
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Stream the file, then delete it once the response is fully sent.
    @after_this_request
    def _cleanup(resp):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return resp

    return send_file(tmp_path, mimetype="application/zip",
                     as_attachment=True, download_name="photo_studio_export.zip")


# --------------------------------------------------------------------------- #
# Frontend (single page, vanilla JS — no build step)
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Photo Studio</title>
<style>
:root{
  --bg:#15140f; --panel:#1f1d16; --panel2:#272419; --line:#3a3527;
  --ink:#ece6d6; --mut:#9a917c; --accent:#e0a23b; --accent2:#7fae6f;
  --warn:#d9694a; --radius:10px;
}
*{box-sizing:border-box}
body{margin:0;background:
  radial-gradient(1200px 600px at 80% -10%,#2a2718 0,transparent 60%),
  var(--bg);
  color:var(--ink);font-family:"Iowan Old Style","Palatino Linotype",Georgia,serif;}
header{display:flex;align-items:center;gap:18px;padding:14px 22px;
  border-bottom:1px solid var(--line);background:rgba(20,19,14,.8);
  position:sticky;top:0;z-index:20;backdrop-filter:blur(6px)}
h1{font-size:20px;margin:0;letter-spacing:.5px}
h1 .dot{color:var(--accent)}
.sub{color:var(--mut);font-size:13px}
.wrap{display:grid;grid-template-columns:300px 1fr;gap:0;min-height:calc(100vh - 56px)}
aside{border-right:1px solid var(--line);padding:18px;background:var(--panel);
  position:sticky;top:56px;height:calc(100vh - 56px);overflow:auto}
main{padding:18px 22px}
.group{margin-bottom:22px}
.group h3{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;
  color:var(--mut);margin:0 0 10px;font-family:ui-monospace,monospace}
label{display:block;font-size:13px;margin:10px 0 4px;color:var(--mut)}
input[type=text],input[type=password],input[type=number]{width:100%;padding:8px 10px;
  background:var(--panel2);border:1px solid var(--line);border-radius:8px;
  color:var(--ink);font-family:ui-monospace,monospace;font-size:13px}
input[type=range]{width:100%;accent-color:var(--accent)}
button{cursor:pointer;border:1px solid var(--line);background:var(--panel2);
  color:var(--ink);padding:9px 14px;border-radius:8px;font-size:13px;
  font-family:inherit;transition:.15s}
button:hover{border-color:var(--accent);color:#fff}
button.primary{background:var(--accent);color:#1a160c;border-color:var(--accent);font-weight:600}
button.primary:hover{filter:brightness(1.08)}
.row{display:flex;gap:8px;align-items:center}
.drop{border:1.5px dashed var(--line);border-radius:var(--radius);padding:18px;
  text-align:center;color:var(--mut);font-size:13px;margin-top:8px}
.drop.over{border-color:var(--accent);color:var(--ink)}
.stat{display:flex;justify-content:space-between;font-size:13px;padding:4px 0;
  border-bottom:1px dotted var(--line)}
.stat b{color:var(--accent)}
.filelist{font-family:ui-monospace,monospace;font-size:11px;color:var(--mut);
  max-height:140px;overflow:auto;margin-bottom:8px}
.filelist .fl{display:flex;justify-content:space-between;gap:6px;padding:2px 0;
  white-space:nowrap}
.filelist .fl .nm{overflow:hidden;text-overflow:ellipsis}
.filelist .fl .ct{color:var(--accent2);flex:0 0 auto}
.filelist .hd{color:var(--ink);text-transform:uppercase;letter-spacing:1px;
  font-size:10px;margin:2px 0 4px}
.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:16px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
  overflow:hidden;display:flex;flex-direction:column;transition:.15s}
.card:hover{border-color:#5a5340}
.card.skip{opacity:.45}
.thumbwrap{position:relative;aspect-ratio:4/3;background:#0d0c09;display:flex;
  align-items:center;justify-content:center;overflow:hidden}
.thumbwrap img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-in}
.lightbox{position:fixed;inset:0;background:rgba(8,7,5,.92);z-index:60;display:none;
  flex-direction:column;padding:14px 18px}
.lightbox.open{display:flex}
.lbhead{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.lbhead h2{font-size:15px;margin:0;font-family:ui-monospace,monospace;color:var(--ink)}
.lbbody{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto;min-height:0}
.lbbody img{max-width:100%;max-height:100%;object-fit:contain;cursor:zoom-out}
.lbmeta{font-size:12px;color:var(--mut);font-family:ui-monospace,monospace}
.badges{position:absolute;top:6px;left:6px;display:flex;gap:4px;flex-wrap:wrap}
.badge{font-family:ui-monospace,monospace;font-size:10px;padding:2px 6px;border-radius:5px;
  background:rgba(0,0,0,.6);color:var(--ink);border:1px solid var(--line)}
.badge.dup{background:rgba(217,105,74,.85);color:#fff;border:0}
.badge.blank{background:rgba(154,145,124,.85);color:#1a160c;border:0}
.badge.grp{color:var(--accent2)}
.cardbody{padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.cardbody .nm{font-size:12px;color:var(--mut);font-family:ui-monospace,monospace;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.desc{font-size:13px;line-height:1.35;min-height:18px}
.tags{display:flex;gap:5px;flex-wrap:wrap}
.tag{font-size:11px;padding:2px 8px;border-radius:20px;background:var(--panel2);
  border:1px solid var(--line);color:var(--ink)}
.meta{font-size:11px;color:var(--mut);font-family:ui-monospace,monospace;
  display:flex;justify-content:space-between}
.cardctl{display:flex;gap:6px;border-top:1px solid var(--line);padding:8px 10px}
.cardctl button{flex:0 0 auto;padding:6px 9px}
.cardctl .grow{flex:1}
.empty{color:var(--mut);text-align:center;padding:80px 0;font-style:italic}
.spin{display:inline-block;width:14px;height:14px;border:2px solid var(--line);
  border-top-color:var(--accent);border-radius:50%;animation:s 0.7s linear infinite}
@keyframes s{to{transform:rotate(360deg)}}
.note{font-size:11px;color:var(--mut);margin-top:6px;line-height:1.4}
.modal{position:fixed;inset:0;background:rgba(10,9,6,.94);z-index:50;display:none;
  flex-direction:column;padding:14px 18px}
.modal.open{display:flex}
.ehead{display:flex;align-items:center;gap:12px;margin-bottom:10px}
.ehead h2{font-size:16px;margin:0}
.ebody{flex:1;overflow:auto;display:flex;justify-content:center;align-items:flex-start}
#cv{background:#0d0c09;border:1px solid var(--line);cursor:crosshair;touch-action:none}
.einfo{font-size:13px;color:var(--mut)}
.efoot{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}
</style></head>
<body>
<header>
  <h1>Photo<span class="dot">.</span>Studio</h1>
  <span class="sub">scan &rarr; crop &rarr; tag &rarr; export</span>
  <span id="status" class="sub" style="margin-left:auto"></span>
</header>
<div class="wrap">
<aside>
  <div class="group">
    <h3>1 · Source</h3>
    <input type="file" id="fileInput" accept="image/*" multiple style="display:none" onchange="uploadFiles(this.files)">
    <button class="primary" style="width:100%" onclick="document.getElementById('fileInput').click()">Choose scan images…</button>
    <div class="drop" id="drop">…or drop scan images here</div>
    <button onclick="clearSession()" style="width:100%;margin-top:8px">New session (clear)</button>
    <button onclick="openEditor()" style="width:100%;margin-top:8px">&#9986; Manual crop (draw boxes)</button>
  </div>
  <!-- Detection / autocrop panel hidden until autocrop is fixed.
       The detect() backend + JS remain; restore this block to re-enable.
  <div class="group">
    <h3>2 · Detection</h3>
    <label>Sensitivity <span id="sv">1.0</span></label>
    <input type="range" id="sens" min="0.4" max="1.8" step="0.05" value="1.0" oninput="sv.textContent=this.value">
    <label>Min photo size (% of scan) <span id="mv">1</span></label>
    <input type="range" id="minarea" min="0.1" max="10" step="0.1" value="1" oninput="mv.textContent=this.value">
    <label>Padding (px)</label>
    <input type="number" id="pad" value="6" min="0" max="60">
    <label class="row" style="gap:8px"><input type="checkbox" id="deskew" checked style="width:auto"> Deskew (rotate upright)</label>
    <button class="primary" style="width:100%;margin-top:12px" onclick="detect()">Detect photos</button>
  </div>
  -->
  <div class="group">
    <h3>2 · Export</h3>
    <button class="primary" style="width:100%" onclick="doDownload()">Download .zip</button>
    <div class="note">Downloads corrected photos, blanks, and manifests (CSV + JSON) to your computer.</div>
  </div>
  <div class="group">
    <h3>Summary</h3>
    <div class="stat"><span>Scans</span><b id="nScans">0</b></div>
    <div class="stat"><span>Photos</span><b id="nPhotos">0</b></div>
    <div class="stat"><span>Blank</span><b id="nBlank">0</b></div>
    <div class="stat"><span>Duplicate copies</span><b id="nDup">0</b></div>
    <div class="stat"><span>Keeping</span><b id="nKeep">0</b></div>
  </div>
  <div class="group">
    <h3>Files</h3>
    <div class="filelist" id="fileScans"></div>
    <div class="filelist" id="fileCrops"></div>
  </div>
</aside>
<main>
  <div class="toolbar">
    <label class="row" style="margin:0;gap:6px"><input type="checkbox" id="showBlank" style="width:auto" onchange="render()"> show blanks</label>
    <label class="row" style="margin:0;gap:6px"><input type="checkbox" id="showDup" style="width:auto" onchange="render()"> show duplicate copies</label>
  </div>
  <div id="grid" class="grid"></div>
  <div id="empty" class="empty">Load or drop scans, then use <b>Manual crop</b> to draw boxes around each photo.</div>
</main>
</div>
<div class="modal" id="editor">
  <div class="ehead">
    <h2>Manual crop — <span id="eScanName"></span></h2>
    <span class="einfo">Drag a box around each photo (corner to opposite corner). Drag a handle to adjust a corner; tap a corner then use arrow keys to nudge (Shift = 10px).</span>
    <button style="margin-left:auto" onclick="closeEditor()">Close</button>
  </div>
  <div class="ebody"><canvas id="cv"></canvas></div>
  <div class="efoot">
    <button onclick="removeLastQuad()">Remove last box</button>
    <span class="einfo" id="eCount"></span>
    <label class="einfo" style="margin-left:10px">skew
      <input type="range" id="skew" min="-15" max="15" step="0.1" value="0"
             style="vertical-align:middle;width:160px" oninput="onSkew(this.value)"
             disabled>
      <span id="skewVal">0.0°</span></label>
    <button onclick="prevScan()">&lsaquo; Prev</button>
    <button onclick="nextScan()">Next &rsaquo;</button>
    <button class="primary" style="margin-left:auto" onclick="cropAll()">Crop all &amp; close</button>
  </div>
</div>
<div class="lightbox" id="lightbox" onclick="closeLightbox(event)">
  <div class="lbhead">
    <h2 id="lbName"></h2>
    <span class="lbmeta" id="lbMeta"></span>
    <button style="margin-left:auto" onclick="closeLightbox(event,true)">Close ✕</button>
  </div>
  <div class="lbbody"><img id="lbImg" src="" alt=""></div>
</div>
<script>
let photos=[], scans=[];
const $=id=>document.getElementById(id);
function setStatus(t,busy){ $("status").innerHTML = busy?'<span class="spin"></span> '+t : t; }
function openLightbox(id){
  const p=photos.find(x=>x.id===id); if(!p)return;
  $("lbImg").src='/api/full/'+p.id+'?r='+p.rotation_cw;
  $("lbName").textContent=p.scan_name+' · #'+p.index;
  const dec=p.decade?(' · '+p.decade):'';
  $("lbMeta").textContent=p.w+'×'+p.h+' px · '+(p.scene_type||'untagged')+dec;
  $("lightbox").classList.add('open');
}
function closeLightbox(e,force){
  // close on backdrop/button/Esc, but not when clicking the image itself
  if(force || !e || e.target.id==='lightbox' || e.target.tagName==='BUTTON'
     || e.target.id==='lbImg'){
    $("lightbox").classList.remove('open'); $("lbImg").src='';
  }
}
document.addEventListener('keydown',e=>{
  if(e.key==='Escape' && $("lightbox").classList.contains('open')) closeLightbox(null,true);
});
async function api(url,body){
  const o={method:'POST',headers:{'Content-Type':'application/json'}};
  if(body)o.body=JSON.stringify(body);
  const r=await fetch(url,o); return r.json();
}
async function uploadFiles(fileList){
  if(!fileList||!fileList.length){return;}
  const fd=new FormData();
  for(const f of fileList) fd.append('files',f);
  setStatus('uploading '+fileList.length+' file(s)…',true);
  try{
    const r=await(await fetch('/api/upload',{method:'POST',body:fd})).json();
    scans=r.scans; $("nScans").textContent=scans.length;
    setStatus(scans.length+' scan(s) uploaded'); renderFiles();
  }catch(e){ setStatus('upload error: '+e); }
}
function detectParams(){
  // Detection panel is hidden; fall back to defaults if its inputs are absent.
  const val=(id,d)=>{const e=$(id); return e?e.value:d;};
  const chk=(id,d)=>{const e=$(id); return e?e.checked:d;};
  return{
    sensitivity:parseFloat(val("sens",1.0)),
    min_area_frac:parseFloat(val("minarea",1))/100,
    pad:parseInt(val("pad",6)), deskew:chk("deskew",true), dup_dist:10};}
async function detect(){
  if(!scans.length){setStatus('load a folder first');return;}
  setStatus('detecting…',true);
  const r=await api('/api/detect',detectParams());
  photos=r.photos; setStatus(photos.length+' photo(s) found'); render();
}
async function rotate(id,dir){
  const p=photos.find(x=>x.id===id); p.rotation_cw=((p.rotation_cw+dir*90)%360+360)%360;
  await api('/api/update',{id,rotation_cw:p.rotation_cw}); render();
}
async function toggleKeep(id){
  const p=photos.find(x=>x.id===id); p.status=p.status==='keep'?'skip':'keep';
  await api('/api/update',{id,status:p.status}); render();
}
async function editTags(id){
  const p=photos.find(x=>x.id===id);
  const v=prompt('Tags (comma-separated):',p.tags.join(', ')); if(v===null)return;
  const r=await api('/api/update',{id,tags:v}); p.tags=r.photo.tags; render();
}
async function editDesc(id){
  const p=photos.find(x=>x.id===id);
  const v=prompt('Description:',p.description||''); if(v===null)return;
  const r=await api('/api/update',{id,description:v}); p.description=r.photo.description; render();
}
async function deleteCrop(id){
  if(!confirm('Delete this crop? Its dimmed region on the scan will clear too.'))return;
  const r=await api('/api/delete_crop',{id});
  if(r.error){setStatus(r.error);return;}
  photos=r.photos; setStatus('crop deleted'); render();
}
async function doDownload(){
  setStatus('building zip…',true);
  try{
    const resp=await fetch('/api/download',{method:'POST',
      headers:{'Content-Type':'application/json'},body:'{}'});
    if(!resp.ok){
      let msg='download failed';
      try{ msg=(await resp.json()).error||msg; }catch(e){}
      setStatus(msg); return;
    }
    const blob=await resp.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url; a.download='photo_studio_export.zip';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    setStatus('downloaded photo_studio_export.zip');
  }catch(e){ setStatus('download error: '+e); }
}
function counts(){
  $("nPhotos").textContent=photos.length;
  $("nBlank").textContent=photos.filter(p=>p.blank).length;
  $("nDup").textContent=photos.filter(p=>p.is_dup_copy).length;
  $("nKeep").textContent=photos.filter(p=>p.status==='keep').length;
  renderFiles();
}
function renderFiles(){
  // Uploaded scans, each with how many crops came from it.
  const fs=$("fileScans"), fc=$("fileCrops");
  if(fs){
    fs.innerHTML='<div class="hd">Uploaded ('+scans.length+')</div>'+
      (scans.length? scans.map(s=>{
        const n=photos.filter(p=>p.scan_id===s.id).length;
        return `<div class="fl"><span class="nm">${s.name}</span><span class="ct">${n} crop${n!==1?'s':''}</span></div>`;
      }).join('') : '<div class="fl"><span class="nm">none yet</span></div>');
  }
  if(fc){
    fc.innerHTML='<div class="hd">Created ('+photos.length+')</div>'+
      (photos.length? photos.map(p=>
        `<div class="fl"><span class="nm">${p.scan_name} · #${p.index}</span><span class="ct">${p.w}×${p.h}</span></div>`
      ).join('') : '<div class="fl"><span class="nm">none yet</span></div>');
  }
}
function render(){
  counts();
  const showB=$("showBlank").checked, showD=$("showDup").checked;
  const vis=photos.filter(p=>(showB||!p.blank)&&(showD||!p.is_dup_copy));
  $("empty").style.display=vis.length?'none':'block';
  $("grid").innerHTML=vis.map(p=>{
    const badges=[`<span class="badge grp">grp ${p.dup_group}</span>`,
      p.rotation_cw?`<span class="badge">↻${p.rotation_cw}°</span>`:'',
      p.is_dup_copy?`<span class="badge dup">dup</span>`:'',
      p.blank?`<span class="badge blank">blank</span>`:''].join('');
    const tags=p.tags.map(t=>`<span class="tag">${t}</span>`).join('');
    return `<div class="card ${p.status==='skip'?'skip':''}">
      <div class="thumbwrap"><img src="/api/thumb/${p.id}?r=${p.rotation_cw}" loading="lazy" onclick="openLightbox('${p.id}')" title="Click to view full size">
        <div class="badges">${badges}</div></div>
      <div class="cardbody">
        <div class="nm">${p.scan_name} · #${p.index}</div>
        <div class="desc">${p.description||'<span style="color:var(--mut)">no description</span>'}</div>
        <div class="tags">${tags}</div>
        <div class="meta"><span>${p.added_by&&p.added_by!=='local'?p.added_by:''}</span><span></span></div>
      </div>
      <div class="cardctl">
        <button onclick="rotate('${p.id}',-1)">↺</button>
        <button onclick="rotate('${p.id}',1)">↻</button>
        <button class="grow" onclick="editDesc('${p.id}')">desc</button>
        <button class="grow" onclick="editTags('${p.id}')">tags</button>
        <button onclick="toggleKeep('${p.id}')">${p.status==='keep'?'keep':'skip'}</button>
        <button onclick="deleteCrop('${p.id}')" title="Delete crop">🗑</button>
      </div></div>`;
  }).join('');
}
// drag & drop upload
const drop=$("drop");
['dragover','dragenter'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>{ uploadFiles(ev.dataTransfer.files); });
async function clearSession(){
  if(!confirm('Clear the current session and delete its cropped files?'))return;
  await api('/api/clear'); photos=[]; scans=[];
  $("nScans").textContent=0; render(); setStatus('session cleared');
}
let lastRev=0;
async function init(){
  try{
    const r=await(await fetch('/api/state')).json();
    scans=r.scans||[]; photos=r.photos||[]; lastRev=r.rev||0;
    $("nScans").textContent=scans.length;
    renderFiles();
    if(photos.length){ setStatus('restored '+photos.length+' photo(s) from last session'); render(); }
  }catch(e){}
  try{
    const c=await(await fetch('/api/config')).json();
    if(c.multiuser){
      $("status").textContent='signed in as '+c.user;
      setInterval(syncShared, 5000);   // shared workspace: pick up others' changes
    }
  }catch(e){}
}
async function syncShared(){
  if($("editor").classList.contains('open'))return;  // don't disturb active editing
  try{
    const r=await(await fetch('/api/rev')).json();
    if((r.rev||0)===lastRev)return;
    const st=await(await fetch('/api/state')).json();
    scans=st.scans||[]; photos=st.photos||[]; lastRev=st.rev||0;
    $("nScans").textContent=scans.length; render();
  }catch(e){}
}
// ---- manual crop editor ----
let ed={i:0, quads:{}, angles:{}, pts:[], img:null, scale:1, drag:null, newbox:null, sel:null};
// Rotate a box's 4 base (axis-aligned) corners about their centre by deg.
function rotatedQuad(base, deg){
  if(!deg) return base.map(p=>[p[0],p[1]]);
  const cx=(base[0][0]+base[2][0])/2, cy=(base[0][1]+base[2][1])/2;
  const r=deg*Math.PI/180, cos=Math.cos(r), sin=Math.sin(r);
  return base.map(([x,y])=>{
    const dx=x-cx, dy=y-cy;
    return [cx+dx*cos-dy*sin, cy+dx*sin+dy*cos];
  });
}
// The corners to draw / send for box qi on the current scan (base + tilt).
function dispQuad(sid, qi){
  const base=ed.quads[sid][qi];
  const deg=(ed.angles[sid]&&ed.angles[sid][qi])||0;
  return rotatedQuad(base, deg);
}
const cv=$("cv"), ctx=cv.getContext('2d');
function openEditor(){
  if(!scans.length){setStatus('load a folder or drop scans first');return;}
  ed.i=0; ed.quads={}; ed.angles={}; ed.pts=[]; $("editor").classList.add('open'); loadEScan();
}
function closeEditor(){ $("editor").classList.remove('open'); }
function curScan(){ return scans[ed.i]; }
function loadEScan(){
  const s=curScan();
  $("eScanName").textContent=s.name+" ("+(ed.i+1)+"/"+scans.length+")";
  ed.pts=[]; ed.sel=null; ed.newbox=null; if(!ed.quads[s.id])ed.quads[s.id]=[]; if(!ed.angles[s.id])ed.angles[s.id]=[];
  if(typeof syncSkewUI==='function') syncSkewUI();
  ed.img=new Image();
  ed.img.onload=()=>{
    const maxW=Math.min(window.innerWidth-60,1000), maxH=window.innerHeight-170;
    ed.scale=Math.min(maxW/ed.img.naturalWidth, maxH/ed.img.naturalHeight, 1);
    cv.width=Math.round(ed.img.naturalWidth*ed.scale);
    cv.height=Math.round(ed.img.naturalHeight*ed.scale);
    redraw();
  };
  ed.img.src='/api/scan/'+s.id+'?t='+Date.now();
}
function eCanvasPos(e){
  const r=cv.getBoundingClientRect();
  return [(e.clientX-r.left)*(cv.width/r.width), (e.clientY-r.top)*(cv.height/r.height)];
}
function eFindCorner(cx,cy){
  const sc=ed.scale, HIT=12, sid=curScan().id, qs=ed.quads[sid]||[];
  for(let qi=0;qi<qs.length;qi++){
    const dq=dispQuad(sid,qi);
    for(let pi=0;pi<4;pi++){
      if(Math.hypot(dq[pi][0]*sc-cx, dq[pi][1]*sc-cy)<=HIT) return {q:qi,p:pi};
    }
  }
  for(let pi=0;pi<ed.pts.length;pi++){
    if(Math.hypot(ed.pts[pi][0]*sc-cx, ed.pts[pi][1]*sc-cy)<=HIT) return {pts:true,p:pi};
  }
  return null;
}
cv.addEventListener('pointerdown',e=>{
  const [cx,cy]=eCanvasPos(e); const hit=eFindCorner(cx,cy);
  if(hit){ ed.drag=hit; ed.sel=hit; }
  else { ed.newbox={x0:cx/ed.scale, y0:cy/ed.scale, x1:cx/ed.scale, y1:cy/ed.scale}; ed.sel=null; }
  syncSkewUI();
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove',e=>{
  const [cx,cy]=eCanvasPos(e);
  if(ed.drag){
    // Rigid rectangle resize: the dragged corner moves to the pointer, the
    // diagonally-opposite corner stays anchored, and the two neighbours are
    // recomputed so the box stays axis-aligned. Order is [TL,TR,BR,BL].
    // Resizing returns the box to axis-aligned (angle reset); apply tilt after.
    const sid=curScan().id, q=ed.quads[sid][ed.drag.q];
    if(ed.angles[sid]) ed.angles[sid][ed.drag.q]=0;
    const i=ed.drag.p, opp=(i+2)%4;
    const ax=q[opp][0], ay=q[opp][1];       // anchored opposite corner
    const nx=cx/ed.scale, ny=cy/ed.scale;    // new dragged corner
    const x0=Math.min(ax,nx), x1=Math.max(ax,nx);
    const y0=Math.min(ay,ny), y1=Math.max(ay,ny);
    q[0]=[x0,y0]; q[1]=[x1,y0]; q[2]=[x1,y1]; q[3]=[x0,y1];
    redraw();
  } else if(ed.newbox){
    ed.newbox.x1=cx/ed.scale; ed.newbox.y1=cy/ed.scale; redraw();
  } else {
    cv.style.cursor = eFindCorner(cx,cy) ? 'move' : 'crosshair';
  }
});
function ePointerUp(e){
  if(ed.drag){ ed.drag=null; syncSkewUI(); return; }
  if(ed.newbox){
    const b=ed.newbox; ed.newbox=null;
    const x0=Math.min(b.x0,b.x1), x1=Math.max(b.x0,b.x1);
    const y0=Math.min(b.y0,b.y1), y1=Math.max(b.y0,b.y1);
    if((x1-x0)>20 && (y1-y0)>20)
      ed.quads[curScan().id].push([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]);
      ed.angles[curScan().id].push(0);
    redraw();
  }
}
cv.addEventListener('pointerup',ePointerUp);
cv.addEventListener('pointercancel',()=>{ed.drag=null; ed.newbox=null;});
function eSelPoint(){
  if(!ed.sel)return null;
  if(ed.sel.pts) return ed.pts[ed.sel.p];
  const sid=curScan().id, q=ed.quads[sid];
  if(!(q&&q[ed.sel.q])) return null;
  return dispQuad(sid, ed.sel.q)[ed.sel.p];   // displayed (rotated) position
}
// Sync the skew slider to the currently-selected box (enable + show its angle).
function syncSkewUI(){
  const s=$("skew"), v=$("skewVal");
  if(ed.sel && !ed.sel.pts){
    const sid=curScan().id, a=(ed.angles[sid]&&ed.angles[sid][ed.sel.q])||0;
    s.disabled=false; s.value=a; v.textContent=a.toFixed(1)+'°';
  } else {
    s.disabled=true; s.value=0; v.textContent='0.0°';
  }
}
// Slider moved: set the selected box's tilt and redraw.
function onSkew(val){
  if(!ed.sel || ed.sel.pts) return;
  const sid=curScan().id, a=parseFloat(val);
  if(!ed.angles[sid]) ed.angles[sid]=[];
  ed.angles[sid][ed.sel.q]=a;
  $("skewVal").textContent=a.toFixed(1)+'°';
  redraw();
}
document.addEventListener('keydown',e=>{
  if(!$("editor").classList.contains('open')||!ed.sel)return;
  const map={ArrowLeft:[-1,0],ArrowRight:[1,0],ArrowUp:[0,-1],ArrowDown:[0,1]};
  if(!(e.key in map))return;
  e.preventDefault();
  const step=e.shiftKey?10:1, d=map[e.key];
  // Nudge a committed box corner: move it, keep the opposite corner anchored,
  // and rebuild as an axis-aligned rectangle (matches rigid drag behaviour).
  if(!ed.sel.pts){
    const q=ed.quads[curScan().id]&&ed.quads[curScan().id][ed.sel.q];
    if(!q){ed.sel=null;return;}
    const i=ed.sel.p, opp=(i+2)%4;
    const ax=q[opp][0], ay=q[opp][1];
    const nx=q[i][0]+d[0]*step, ny=q[i][1]+d[1]*step;
    const x0=Math.min(ax,nx), x1=Math.max(ax,nx);
    const y0=Math.min(ay,ny), y1=Math.max(ay,ny);
    q[0]=[x0,y0]; q[1]=[x1,y0]; q[2]=[x1,y1]; q[3]=[x0,y1];
    redraw(); return;
  }
  const pt=eSelPoint(); if(!pt){ed.sel=null;return;}
  pt[0]+=d[0]*step; pt[1]+=d[1]*step; redraw();
});
function redraw(){
  if(!ed.img)return;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(ed.img,0,0,cv.width,cv.height);
  const sc=ed.scale, sid=curScan().id, qs=ed.quads[sid]||[];
  // Dim regions already cropped from THIS scan (live from stored origin_quads).
  // Deleting a crop removes its photo, so its dim disappears on next redraw.
  const done=(typeof photos!=='undefined'?photos:[]).filter(
    p=>p.scan_id===sid && Array.isArray(p.origin_quad) && p.origin_quad.length===4);
  if(done.length){
    ctx.save();
    done.forEach(p=>{
      ctx.beginPath();
      p.origin_quad.forEach((pt,k)=>{const X=pt[0]*sc,Y=pt[1]*sc; k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
      ctx.closePath();
      ctx.fillStyle='rgba(8,7,5,.62)';     // darken the already-used area
      ctx.fill();
      ctx.strokeStyle='rgba(224,162,59,.55)'; ctx.lineWidth=1.5; ctx.stroke();
    });
    ctx.restore();
  }
  ctx.lineWidth=3; ctx.font='bold 20px ui-monospace,monospace';
  qs.forEach((base,idx)=>{
    const q=dispQuad(sid,idx);
    ctx.strokeStyle='#7fae6f'; ctx.fillStyle='rgba(127,174,111,.18)';
    ctx.beginPath();
    q.forEach((p,k)=>{const X=p[0]*sc,Y=p[1]*sc; k?ctx.lineTo(X,Y):ctx.moveTo(X,Y);});
    ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle='#7fae6f';
    q.forEach(p=>{ctx.beginPath(); ctx.arc(p[0]*sc,p[1]*sc,6,0,7); ctx.fill();});
    ctx.fillStyle='#cfe8c4'; ctx.fillText(idx+1, q[0][0]*sc+8, q[0][1]*sc-8);
  });
  ctx.fillStyle='#e0a23b'; ctx.strokeStyle='#e0a23b';
  if(ed.newbox){
    const b=ed.newbox;
    ctx.lineWidth=2; ctx.setLineDash([8,5]);
    ctx.strokeRect(Math.min(b.x0,b.x1)*sc, Math.min(b.y0,b.y1)*sc,
                   Math.abs(b.x1-b.x0)*sc, Math.abs(b.y1-b.y0)*sc);
    ctx.setLineDash([]);
  }
  const ssp=eSelPoint();
  if(ssp){ ctx.strokeStyle='#fff'; ctx.lineWidth=2.5;
    ctx.beginPath(); ctx.arc(ssp[0]*sc, ssp[1]*sc, 10, 0, 7); ctx.stroke(); }
  const total=Object.values(ed.quads).reduce((a,q)=>a+q.length,0);
  $("eCount").textContent=qs.length+" box(es) here · "+total+" total";
}
function undoPoint(){
  if(ed.pts.length)ed.pts.pop();
  else{const sid=curScan().id, q=ed.quads[sid];
       if(q&&q.length){q.pop(); if(ed.angles[sid])ed.angles[sid].pop();}}
  ed.sel=null; redraw();
}
function removeLastQuad(){ const sid=curScan().id, q=ed.quads[sid];
  if(q&&q.length){q.pop(); if(ed.angles[sid])ed.angles[sid].pop();} ed.sel=null; redraw(); }
function prevScan(){ if(ed.i>0){ed.i--; loadEScan();} }
function nextScan(){ if(ed.i<scans.length-1){ed.i++; loadEScan();} }
async function cropAll(){
  setStatus('cropping…',true);
  let n=0;
  for(const sid of Object.keys(ed.quads)){
    const base=ed.quads[sid]; if(!base.length)continue;
    const quads=base.map((_,qi)=>dispQuad(sid,qi));   // apply per-box tilt
    const r=await api('/api/crop_manual',{scan_id:sid,quads}); n+=r.added||0;
  }
  closeEditor();
  const st=await(await fetch('/api/state')).json(); photos=st.photos||[]; render();
  setStatus('cropped '+n+' photo(s)');
}
init();
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}"
    print(f"Photo Studio → {url}   (work dir: {WORK})")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, threaded=True)


# Load persisted session at import time too, so production servers (gunicorn,
# which import `photo_studio:app` and never call main()) restore state.
load_session()


if __name__ == "__main__":
    main()
