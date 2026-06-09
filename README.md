# Photo Studio — scan splitting & tagging

A local web app for turning flatbed scans (several photos per sheet) into
individual, deskewed, optionally auto-tagged photos. Runs entirely on your
machine; nothing is uploaded except the optional Claude tagging call.

## Quick start

1. Keep all the files below in one folder.
2. Double-click the launcher for your OS:
   - macOS: **Photo Studio.command** (first time: right-click → Open)
   - Windows: **Photo Studio.bat**
   - Linux: **launch.sh**
   The launcher installs any missing packages, picks a free port, starts the
   app, and opens your browser. (Or run `python launch.py` directly.)

Requirements: Python 3, plus `flask`, `opencv-python`, `numpy` (auto-installed
by the launcher on first run). Optional tagging needs an Anthropic API key.

## Using it

1. **Source** — enter a folder of scans or drag images in.
2. **Manual crop (draw boxes)** — the reliable path for tightly packed or
   faded prints. Drag a box around each photo (press at one corner, release at
   the opposite). Drag any corner handle to adjust it, or tap a corner and use
   the arrow keys to nudge (Shift = 10px) — including pulling a corner off-square
   to deskew a tilted photo. Each box is perspective-warped into its own upright
   photo.
3. **Detection** — automatic cropping. Works well only when photos sit on a
   clear/dark background with gaps; faded prints abutting on a light bed are
   unreliable (use Manual crop instead).
4. **Auto-tag** — optional. Adds description, tags, scene type, people count,
   and an estimated decade. Two backends, chosen on the server:
   - **Anthropic** (`PHOTOSTUDIO_TAGGER=anthropic`): paste an API key in the
     sidebar or set `ANTHROPIC_API_KEY`; set a current vision model.
   - **Ollama** (`PHOTOSTUDIO_TAGGER=ollama`): set `OLLAMA_URL` and
     `OLLAMA_MODEL` (e.g. `gemma4`) — runs a local/remote open vision model.
   Blanks and duplicate copies are skipped.
5. **Export** — writes `corrected/` (kept photos, rotation applied),
   `blank/` (set-aside), and `manifest.csv` / `manifest.json`.

Per photo you also get rotate buttons, an editable tag list, and keep/skip.
Work persists across restarts in `~/.photostudio`; "New session" clears it.

## Files

- `photo_studio.py` — the web app (UI + server).
- `scan_splitter.py` — CV helpers for background/mask/deskew (auto-detect).
- `photo_tagger.py` — blank/duplicate detection, orientation hint, tagging.
- `launch.py` — cross-platform launcher.
- `Photo Studio.command` / `Photo Studio.bat` / `launch.sh` — OS double-click wrappers.
- `test_editor.py` — headless-browser regression test for the crop editor
  (`pip install playwright && python -m playwright install chromium`).
- `DEPLOY.md` — deploy to a server (gunicorn + nginx + HTTPS + login) with
  tagging via a remote Ollama over Tailscale.

## Deployment

For running this as a real web app (e.g. a DigitalOcean droplet) with the
vision model on your own machine, see **DEPLOY.md**. Set `PHOTOSTUDIO_PASSWORD`
to require login; serve with `gunicorn -w 1 --threads 8 photo_studio:app` behind
nginx + TLS.

## Notes / limits

- Best results for future batches: leave small gaps between photos and put a
  dark sheet behind them when scanning — that makes auto-detection reliable.
- The face-based orientation hint is unreliable on faded/angled photos, so it
  is stored but never auto-applied; set final orientation with the rotate
  buttons (or let Claude suggest it during tagging).
