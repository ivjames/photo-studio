#!/usr/bin/env python3
"""
photo_tagger.py — enrich cropped photos with metadata.

Stage 2 of the scan pipeline. Takes the individual photos produced by
scan_splitter.py and adds:
  * blank / near-blank flagging          (pure CV, free, offline)
  * duplicate grouping                    (perceptual hash, free, offline)
  * upright-orientation correction        (offline face hint + Claude vision)
  * content tags, scene type, people count, estimated decade   (Claude vision)

Design: the cheap deterministic work (blank, duplicate, a face-based
orientation hint) runs in OpenCV with no API cost. Only the semantic calls
(tags, dating, final orientation) go to the vision model, and blanks +
duplicate copies are skipped so you don't pay to analyze them.

Outputs:
  OUTPUT/corrected/   orientation-corrected copies of the kept photos
  OUTPUT/blank/       photos flagged blank (moved aside, not tagged)
  OUTPUT/manifest.csv / manifest.json   all metadata, one row per input photo

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python photo_tagger.py cropped/ -o tagged/

    # CV only, no API key needed (blank + duplicate + face-hint rotation):
    python photo_tagger.py cropped/ -o tagged/ --offline
"""

import argparse
import base64
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_MODEL = "claude-sonnet-4-6"  # verify current string at docs.claude.com


# --------------------------------------------------------------------------- #
# Pure-CV analysis (free, deterministic, offline)
# --------------------------------------------------------------------------- #
def blankness(gray):
    """Return (is_blank, std, edge_density). Blank = flat and edgeless."""
    std = float(gray.std())
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    is_blank = std < 12.0 and edge_density < 0.004
    return is_blank, std, edge_density


def dhash(gray, hash_size=8):
    """64-bit difference hash for near-duplicate detection."""
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a, b):
    return bin(a ^ b).count("1")


def group_duplicates(hashes, max_dist):
    """Greedy union of images whose hashes are within max_dist."""
    n = len(hashes)
    group = list(range(n))

    def find(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if hamming(hashes[i], hashes[j]) <= max_dist:
                group[find(j)] = find(i)
    # Normalize to 0-based group ids.
    roots = {}
    out = []
    for i in range(n):
        r = find(i)
        out.append(roots.setdefault(r, len(roots)))
    return out


# Haar cascade is fast and free; used only as an orientation *hint*.
_FACE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE,
              180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def face_orientation_hint(gray):
    """Try all 4 rotations; return the CW rotation that yields the most/biggest
    upright frontal faces, or None if no faces are found at any rotation."""
    best_rot, best_score = None, 0.0
    for cw, op in _ROTATIONS.items():
        g = gray if op is None else cv2.rotate(gray, op)
        faces = _FACE.detectMultiScale(g, scaleFactor=1.1, minNeighbors=5,
                                       minSize=(30, 30))
        score = sum(w * h for (x, y, w, h) in faces)
        if score > best_score:
            best_rot, best_score = cw, score
    return best_rot  # 0/90/180/270 or None


# --------------------------------------------------------------------------- #
# Claude vision analysis (semantic; tags, dating, orientation)
# --------------------------------------------------------------------------- #
VISION_PROMPT = """You are cataloguing a scanned personal/family photograph.
{hint}
Respond with ONLY a JSON object (no prose, no markdown fences) with keys:
  "description": one short sentence describing the photo
  "tags": array of 3-8 lowercase keywords (subjects, setting, objects)
  "scene_type": one of "portrait","group","landscape","building","event","document","object","other"
  "people_count": integer estimate (0 if none)
  "estimated_decade": best guess like "1970s" or "unknown"
  "decade_confidence": "low","medium", or "high"
  "decade_reasoning": brief cue you used (clothing, film tint, car, etc.)
  "correct_rotation_cw": clockwise degrees (0,90,180,270) to make it upright
  "orientation_reasoning": what told you which way is up (faces, horizon, text)
"""


def encode_image(bgr, max_dim):
    """Downscale (to control tokens/cost) and JPEG-encode to base64."""
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("ascii")


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    s, e = text.find("{"), text.rfind("}")
    return json.loads(text[s:e + 1])


def analyze_with_claude(client, model, bgr, face_hint, max_dim, retries=4):
    hint = (f"A face-detector suggests the upright rotation is {face_hint} deg "
            f"clockwise; confirm or override it." if face_hint is not None
            else "No faces were detected automatically; use horizon, text, or "
                 "other cues to decide orientation.")
    b64 = encode_image(bgr, max_dim)
    msg = [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64",
         "media_type": "image/jpeg", "data": b64}},
        {"type": "text", "text": VISION_PROMPT.format(hint=hint)},
    ]}]
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = client.messages.create(model=model, max_tokens=600, messages=msg)
            text = "".join(b.text for b in resp.content if b.type == "text")
            return parse_json(text)
        except Exception as e:  # noqa: BLE001 — surface, back off, retry
            if attempt == retries - 1:
                return {"error": str(e)}
            time.sleep(delay)
            delay *= 2


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def apply_rotation(bgr, cw):
    op = _ROTATIONS.get(int(cw) % 360 if cw is not None else 0)
    return bgr if op is None else cv2.rotate(bgr, op)


def main():
    ap = argparse.ArgumentParser(description="Tag, date, dedupe and orient cropped photos.")
    ap.add_argument("input", help="Directory of cropped photos (stage-1 output).")
    ap.add_argument("-o", "--output", required=True, help="Output directory.")
    ap.add_argument("--offline", action="store_true",
                    help="Skip Claude; run only CV (blank, duplicate, face-hint rotation).")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Vision model id (default {DEFAULT_MODEL}; confirm at docs.claude.com).")
    ap.add_argument("--max-dim", type=int, default=1024,
                    help="Longest edge sent to the model, px (lower = cheaper).")
    ap.add_argument("--dup-dist", type=int, default=10,
                    help="Max hamming distance for two photos to count as duplicates.")
    ap.add_argument("--no-face-hint", action="store_true", help="Disable the face-detection hint.")
    args = ap.parse_args()

    paths = list_images(args.input)
    if not paths:
        print(f"No images found in {args.input}", file=sys.stderr)
        sys.exit(1)

    out = Path(args.output)
    (out / "corrected").mkdir(parents=True, exist_ok=True)
    (out / "blank").mkdir(parents=True, exist_ok=True)

    client = None
    if not args.offline:
        try:
            from anthropic import Anthropic
        except ImportError:
            print("anthropic SDK not installed. `pip install anthropic` or use --offline.",
                  file=sys.stderr)
            sys.exit(1)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ANTHROPIC_API_KEY not set. Export it or use --offline.", file=sys.stderr)
            sys.exit(1)
        client = Anthropic()

    # Pass 1: load + CV analysis.
    records, hashes, grays, imgs = [], [], [], []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"  ! unreadable: {p}", file=sys.stderr)
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        is_blank, std, edge = blankness(gray)
        rec = {"file": p.name, "path": str(p), "blank": is_blank,
               "std": round(std, 2), "edge_density": round(edge, 5),
               "duplicate_group": None, "is_dup_copy": False}
        records.append(rec)
        hashes.append(dhash(gray))
        grays.append(gray)
        imgs.append(bgr)

    # Duplicate grouping; keep the first member of each group as the "original".
    groups = group_duplicates(hashes, args.dup_dist)
    seen = set()
    for rec, g in zip(records, groups):
        rec["duplicate_group"] = g
        rec["is_dup_copy"] = g in seen
        seen.add(g)

    # Pass 2: orient + tag the keepers.
    for rec, gray, bgr in zip(records, grays, imgs):
        if rec["blank"]:
            shutil.copy2(rec["path"], out / "blank" / rec["file"])
            continue

        face_hint = None if args.no_face_hint else face_orientation_hint(gray)
        rec["face_hint_cw"] = face_hint

        if client and not rec["is_dup_copy"]:
            meta = analyze_with_claude(client, args.model, bgr, face_hint, args.max_dim)
            rec.update({f"ai_{k}": v for k, v in meta.items()})
            rot = meta.get("correct_rotation_cw", face_hint or 0)
        else:
            rot = face_hint or 0  # offline or duplicate copy: trust the face hint
        rec["applied_rotation_cw"] = int(rot) if isinstance(rot, (int, float)) else 0

        corrected = apply_rotation(bgr, rec["applied_rotation_cw"])
        cv2.imwrite(str(out / "corrected" / rec["file"]), corrected)
        tags = rec.get("ai_tags", "")
        print(f"  {rec['file']}: rot={rec['applied_rotation_cw']} "
              f"dupgrp={rec['duplicate_group']} "
              f"{'[dup]' if rec['is_dup_copy'] else ''} {tags}")

    # Manifests.
    with open(out / "manifest.json", "w") as f:
        json.dump(records, f, indent=2, default=str)
    keys = sorted({k for r in records for k in r})
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: (json.dumps(r[k]) if isinstance(r.get(k), (list, dict)) else r.get(k, ""))
                        for k in keys})

    n_blank = sum(r["blank"] for r in records)
    n_groups = len(set(groups))
    n_dups = sum(r["is_dup_copy"] for r in records)
    print(f"\nDone. {len(records)} photos | {n_blank} blank | "
          f"{n_groups} unique groups ({n_dups} duplicate copies).")
    print(f"Corrected images: {out/'corrected'}/  Manifest: {out/'manifest.csv'}")


if __name__ == "__main__":
    main()
