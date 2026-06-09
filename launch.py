#!/usr/bin/env python3
"""
launch.py — one-click launcher for Photo Studio.

Double-click the wrapper for your OS (Photo Studio.command / .bat / launch.sh),
or run `python launch.py` directly. It checks dependencies (installing any that
are missing), picks a free port, starts the app, and opens your browser.
"""

import importlib.util
import os
import socket
import subprocess
import sys
from pathlib import Path

REQUIRED = {"flask": "flask", "cv2": "opencv-python", "numpy": "numpy"}


def ensure_deps():
    missing = [pkg for mod, pkg in REQUIRED.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    print("Installing missing packages: " + ", ".join(missing))
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    except Exception as e:  # noqa: BLE001
        print(f"\nCould not auto-install ({e}).")
        print("Please run this once, then relaunch:")
        print(f"    {sys.executable} -m pip install {' '.join(missing)}")
        input("\nPress Enter to exit…")
        sys.exit(1)


def free_port():
    for p in (5000, 5001, 5050, 8000, 8080):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    with socket.socket() as s:        # fall back to any free port
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    here = Path(__file__).resolve().parent
    os.chdir(here)
    sys.path.insert(0, str(here))

    needed = ["photo_studio.py", "scan_splitter.py", "photo_tagger.py"]
    missing_files = [f for f in needed if not (here / f).exists()]
    if missing_files:
        print("Missing required files next to launch.py: " + ", ".join(missing_files))
        input("Press Enter to exit…")
        sys.exit(1)

    ensure_deps()
    port = free_port()
    print(f"Launching Photo Studio on port {port} … (close this window to quit)")
    sys.argv = ["photo_studio", "--port", str(port)]
    import photo_studio
    photo_studio.main()


if __name__ == "__main__":
    main()
