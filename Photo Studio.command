#!/bin/bash
# Double-click in Finder to start Photo Studio.
# First time only: right-click → Open (to get past Gatekeeper).
cd "$(dirname "$0")"
echo "Starting Photo Studio — your browser will open shortly."
echo "Keep this window open while you work; close it to quit."
python3 launch.py
