# Deploying Photo Studio to a DigitalOcean droplet (tagging via home Ollama)

Architecture: the **app** runs on a small DO droplet (public, behind nginx +
HTTPS + a login). The **vision model** runs in Ollama on your **home machine**,
reachable from the droplet privately over **Tailscale** — so Ollama is never
exposed to the open internet (it has no auth of its own).

```
browser ──HTTPS──> nginx ──> gunicorn (photo_studio) ──Tailscale──> Ollama @ home (gemma4)
                    (droplet)                                         (your PC/GPU)
```

A basic CPU droplet is fine for the app itself. Do **not** run a vision model on
a cheap droplet — it's slow on CPU and the larger sizes won't fit; keep Ollama at
home where you have the RAM/GPU.

---

## 1. Home machine — Ollama + Tailscale

```bash
# install Ollama, then pull a vision model
ollama pull gemma4                  # vision-capable; or gemma3:4b / gemma3:12b

# let Ollama listen on the Tailscale interface (not just localhost)
# Linux (systemd): sudo systemctl edit ollama  ->  add:
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
# macOS: launchctl setenv OLLAMA_HOST 0.0.0.0:11434  (then restart Ollama)

# install Tailscale and join your tailnet
tailscale up
tailscale ip -4                     # note this address, e.g. 100.x.y.z
```

Verify from the home machine: `curl http://localhost:11434/api/tags`.

## 2. Droplet — base setup

```bash
# Ubuntu 24.04 droplet
sudo apt update && sudo apt install -y python3-venv nginx libglib2.0-0
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up                   # join the SAME tailnet
# confirm the droplet can reach home Ollama:
curl http://100.x.y.z:11434/api/tags
```

## 3. App

```bash
sudo mkdir -p /opt/photostudio && cd /opt/photostudio
# copy photo_studio.py, scan_splitter.py, photo_tagger.py here
python3 -m venv venv && . venv/bin/activate
pip install flask gunicorn opencv-python-headless numpy   # headless = no GUI libs
```

Note: `opencv-python-headless` (not `opencv-python`) on a server — it skips the
desktop/GUI dependencies.

## 4. systemd service

`/etc/systemd/system/photostudio.service`:

```ini
[Unit]
Description=Photo Studio
After=network-online.target tailscaled.service

[Service]
WorkingDirectory=/opt/photostudio
Environment="PHOTOSTUDIO_TAGGER=ollama"
Environment="OLLAMA_URL=http://100.x.y.z:11434"   # home machine's Tailscale IP
Environment="OLLAMA_MODEL=gemma4"
Environment="PHOTOSTUDIO_USERS=alice:long-pass-1,bob:long-pass-2"
Environment="PHOTOSTUDIO_HOME=/opt/photostudio/data"
ExecStart=/opt/photostudio/venv/bin/gunicorn -w 1 --threads 8 \
          --timeout 600 -b 127.0.0.1:8000 photo_studio:app
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now photostudio
```

**Use `-w 1` (one worker).** The app keeps session state in memory + on disk;
multiple workers would each have their own copy. One worker with threads is right
for a personal deployment. `--timeout 600` allows slow home-CPU inference.

## 5. nginx + HTTPS

`/etc/nginx/sites-available/photostudio`:

```nginx
server {
    server_name photos.example.com;
    client_max_body_size 100M;          # scans can be large
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_read_timeout 600s;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/photostudio /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d photos.example.com    # free TLS
```

## 6. Use it

Visit `https://photos.example.com` and log in. Each person in
`PHOTOSTUDIO_USERS` gets their own credentials but they all share **one
workspace** — the same scans and cropped photos. When one person crops or
tags, others see it appear within a few seconds (the page polls a revision
counter), and each photo is stamped with who added it. The app shows
"Auto-tag · Ollama (gemma4)" when the Ollama backend is active, and tag
requests travel droplet → Tailscale → your home Ollama.

(A single `PHOTOSTUDIO_USER` + `PHOTOSTUDIO_PASSWORD` pair also works if you
just want one account.)

---

## Switching back to Anthropic
Set `PHOTOSTUDIO_TAGGER=anthropic` and `ANTHROPIC_API_KEY=...` (or paste a key in
the UI). The same app supports either backend.

## Limits / notes
- **Shared workspace, run one worker (`-w 1`).** All accounts share one
  in-memory + on-disk workspace, so a single gunicorn worker is required;
  multiple workers wouldn't share it. This is fine for a handful of people
  editing together. Many concurrent editors or horizontal scaling would need a
  shared datastore (e.g. SQLite/Postgres + object storage) — a larger change.
- Tagging is only as fast as your home machine; a GPU there helps a lot.
- Keep passwords long; they're the only thing in front of a public URL.
