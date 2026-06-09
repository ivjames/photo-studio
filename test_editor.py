#!/usr/bin/env python3
"""
test_editor.py — headless browser regression test for Photo Studio's crop editor.

Drives the real app in headless Chromium (no visible window): places a box,
selects a corner, nudges it with the arrow keys, and drags a corner — asserting
the coordinates change exactly as expected. Generates its own throwaway scan,
so it needs no test data.

Setup (one time):
    pip install playwright
    python -m playwright install chromium

Run:
    python test_editor.py        # exits 0 on pass, 1 on failure
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

PORT = 5099
BASE = f"http://127.0.0.1:{PORT}"


def make_scan(folder):
    img = np.full((2048, 1727, 3), 230, np.uint8)
    rng = np.random.default_rng(0)
    img[100:1000, 100:1600] = rng.integers(0, 255, (900, 1500, 3), dtype=np.uint8)
    cv2.imwrite(str(Path(folder) / "scan.jpg"), img)


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"content-type": "application/json"})
    return urllib.request.urlopen(req).read()


def main():
    work = tempfile.mkdtemp(prefix="ps_test_")
    scans = tempfile.mkdtemp(prefix="ps_scans_")
    make_scan(scans)
    env = dict(os.environ, PHOTOSTUDIO_HOME=work)
    srv = subprocess.Popen([sys.executable, "photo_studio.py", "--port", str(PORT), "--no-browser"],
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failures = []
    try:
        time.sleep(3)
        post("/api/load_folder", {"path": scans})
        with sync_playwright() as p:
            br = p.chromium.launch(args=["--no-sandbox"])
            pg = br.new_page()
            pg.goto(BASE)
            pg.wait_for_function("typeof scans!=='undefined' && scans.length>0", timeout=8000)
            pg.evaluate("openEditor()")
            pg.wait_for_function(
                "ed.img && ed.img.complete && ed.img.naturalWidth>0 "
                "&& document.getElementById('cv').width>320", timeout=10000)
            geo = pg.evaluate("()=>{const r=document.getElementById('cv').getBoundingClientRect();"
                              "return {left:r.left, top:r.top, scale:ed.scale};}")
            L, T, sc = geo["left"], geo["top"], geo["scale"]

            # draw a box by dragging from one corner to the opposite corner
            pg.mouse.move(L + 200 * sc, T + 200 * sc); pg.mouse.down()
            pg.mouse.move(L + 1200 * sc, T + 1500 * sc, steps=8); pg.mouse.up()
            pg.wait_for_timeout(60)
            if pg.evaluate("(ed.quads[curScan().id]||[]).length") != 1:
                failures.append("dragging a box did not create exactly one box")

            pg.mouse.click(L + 200 * sc, T + 200 * sc); pg.wait_for_timeout(60)
            if not pg.evaluate("ed.sel"):
                failures.append("clicking a corner did not select it")

            before = pg.evaluate("eSelPoint().slice()")
            pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(40)
            a1 = pg.evaluate("eSelPoint().slice()")
            pg.keyboard.press("Shift+ArrowDown"); pg.wait_for_timeout(40)
            a2 = pg.evaluate("eSelPoint().slice()")
            if abs((a1[0] - before[0]) - 1) > 1e-6:
                failures.append(f"ArrowRight should move x by 1, moved {a1[0]-before[0]}")
            if abs((a2[1] - a1[1]) - 10) > 1e-6:
                failures.append(f"Shift+ArrowDown should move y by 10, moved {a2[1]-a1[1]}")

            pg.mouse.move(L + 1200 * sc, T + 200 * sc); pg.mouse.down()
            pg.mouse.move(L + 1300 * sc, T + 260 * sc, steps=5); pg.mouse.up()
            pg.wait_for_timeout(60)
            dx, dy = pg.evaluate("ed.quads[curScan().id][0][1]")
            if abs(dx - 1300) > 15 or abs(dy - 260) > 15:
                failures.append(f"drag landed at ({dx:.0f},{dy:.0f}), expected ~(1300,260)")
            br.close()
    finally:
        srv.terminate()

    if failures:
        print("FAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("All editor tests passed (placement, selection, nudge 1px/10px, drag).")


if __name__ == "__main__":
    main()
