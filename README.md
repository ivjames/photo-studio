# Photo Studio — scan splitting & tagging

A web app for turning flatbed scans (several photos per sheet) into individual,
deskewed, tagged photos. Self-hosted; runs behind your own server.

## Run it

```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
python photo_studio.py            # http://127.0.0.1:5000
# production: gunicorn -w 1 —threads 8 photo_studio:app
```

Requires Python 3.9+. For a real deployment (gunicorn + reverse proxy + login),
see **DEPLOY.md**. Set `PHOTOSTUDIO_USERS=“alice:pw,bob:pw”` to require login.

## Using it

1. **Source** — enter a folder of scans or drag images in. The sidebar lists
   uploaded scans and the crops created from each.
1. **Crop scans** — opens a grid of your uploaded scans; each thumbnail shows
   a badge with how many photos you’ve already cropped from it. Click a scan to
   open the crop editor, where you drag a box around every photo (press at one
   corner, release at the opposite). Drag a corner handle to resize (it stays a
   rigid rectangle), or tap a corner and nudge with the arrow keys (Shift =
   10px). **Zoom** in for precise edges. Regions you’ve already cropped are
   dimmed. Prev/Next move through the other scans; closing returns to the scan
   grid. Each box is perspective-warped into its own upright photo.
1. **Review crops** — the main grid; drag the thumb-size slider to taste.
   Click any crop to open it full-size, where a **tilt** slider straightens it
   by re-cropping cleanly from the parent scan (no corner gaps). Per crop you
   get rotate, delete, and a description/tags editor.
1. **Folders** — create folders in the sidebar and drag any crop onto one to
   label it (a crop can be in several). Click a folder to filter the grid;
   “Unfiled” shows crops with no folder. On export, tick *organize into folder
   subdirectories* to mirror folders as `corrected/<Folder>/` (a crop in two
   folders is copied into both; unfiled crops go to `_unfiled`); leave it
   unticked for a flat export. **Download full backup** saves a zip of the
   entire workspace (scans, crops, folders, tags) you can keep or restore from.
1. **Select & batch** — tick the checkbox on any crops to reveal an action bar:
   add/remove a folder, set description/tags, download just those as a zip, or
   delete — all in one go.
1. **Export** — download a `.zip` (corrected photos, blanks, manifests) to your
   computer.

Work persists across restarts in `~/.photostudio`; “New session” clears it.

<!— Automatic detection/autocrop is temporarily hidden in the UI pending a fix;
     the scan_splitter backend remains. AI auto-tagging (description/tags via a
     vision model) is also disabled — tags and descriptions are entered manually.
     The vision backend code is retained but has no UI trigger. —>

## Files

- `photo_studio.py` — the web app (UI + server); the whole app runs from here.
- `scan_splitter.py` — CV helpers (imported by the app; powers the hidden auto-detect).
- `photo_tagger.py` — vision-tagging helpers (imported by the app; tagging is manual in the UI).
- `requirements.txt` — Python dependencies.
- `DEPLOY.md` — deploy to a server (gunicorn behind a reverse proxy, HTTPS, login).

## Deployment

For running this as a real web app (e.g. a DigitalOcean droplet) with the
vision model on your own machine, see **DEPLOY.md**. Set `PHOTOSTUDIO_PASSWORD`
to require login; serve with `gunicorn -w 1 —threads 8 photo_studio:app` behind
nginx + TLS.

## Notes / limits

- Automatic detection is temporarily hidden; cropping is manual for now. When
  it returns, best results come from leaving small gaps between photos and
  putting a dark sheet behind them when scanning.
- Set final orientation with the rotate buttons; straighten tilt with the tilt
  slider in the full-size crop view (it re-crops from the original scan).