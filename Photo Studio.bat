@echo off
cd /d "%~dp0"
echo Starting Photo Studio - your browser will open shortly.
echo Keep this window open while you work; close it to quit.
python launch.py
if errorlevel 1 (
  echo.
  echo Could not start. Make sure Python 3 is installed and on your PATH.
  pause
)
