#!/usr/bin/env bash
# setup.sh — install / update Photo Studio on an Ubuntu 24 server.
# Run by deploy.py over SSH. Safe to re-run; re-run = update.
# Usage: bash setup.sh /tmp/photostudio.env

set -euo pipefail

ENV_FILE="${1:?No env file passed. Run via deploy.py.}"
APP_DIR="/opt/photostudio"
DATA_DIR="$APP_DIR/data"
VENV="$APP_DIR/venv"
SVCFILE="/etc/systemd/system/photostudio.service"
NGINXCONF="/etc/nginx/sites-available/photostudio"

# --- load deploy config -------------------------------------------------------
# shellcheck source=/dev/null
source "$ENV_FILE"
rm -f "$ENV_FILE"     # don't leave credentials on disk

# convenience defaults
DEPLOY_DOMAIN="${DEPLOY_DOMAIN:-}"
DEPLOY_EMAIL="${DEPLOY_EMAIL:-}"

log() { echo ""; echo "==> $*"; }

# --- system packages ----------------------------------------------------------
log "Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -q \
    nginx \
    python3-venv \
    certbot \
    python3-certbot-nginx \
    libglib2.0-0 \
    ufw \
    curl

# --- Tailscale (idempotent) ---------------------------------------------------
if ! command -v tailscale &>/dev/null; then
    log "Installing Tailscale"
    curl -fsSL https://tailscale.com/install.sh | sh
    echo "  *** Run 'tailscale up' on this server to join your tailnet ***"
else
    echo "  tailscale already present"
fi

# --- app files ----------------------------------------------------------------
log "App files"
mkdir -p "$APP_DIR" "$DATA_DIR"
cp /tmp/photo_studio.py /tmp/scan_splitter.py /tmp/photo_tagger.py "$APP_DIR/"

# --- python venv + deps -------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
    log "Creating venv"
    python3 -m venv "$VENV"
fi
log "Python deps"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q flask gunicorn "opencv-python-headless>=4.5" numpy

# --- systemd service ----------------------------------------------------------
log "systemd service"
cat > "$SVCFILE" << EOF
[Unit]
Description=Photo Studio
After=network-online.target

[Service]
WorkingDirectory=$APP_DIR
EnvironmentFile=/etc/photostudio.env
ExecStart=$VENV/bin/gunicorn -w 1 --threads 8 --timeout 600 \\
    -b 127.0.0.1:8000 photo_studio:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# write env to /etc/photostudio.env (permissions 600 — no world-read)
cat > /etc/photostudio.env << EOF
PHOTOSTUDIO_TAGGER="${PHOTOSTUDIO_TAGGER:-ollama}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma4}"
PHOTOSTUDIO_USERS="${PHOTOSTUDIO_USERS:-admin:changeme}"
PHOTOSTUDIO_HOME="$DATA_DIR"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
EOF
chmod 600 /etc/photostudio.env

systemctl daemon-reload
if systemctl is-active --quiet photostudio; then
    systemctl restart photostudio
    echo "  service restarted"
else
    systemctl enable --now photostudio
    echo "  service enabled and started"
fi

# --- nginx --------------------------------------------------------------------
log "nginx"
if [[ -n "$DEPLOY_DOMAIN" ]]; then
    SERVER_NAME="$DEPLOY_DOMAIN"
else
    SERVER_NAME="_"
fi

cat > "$NGINXCONF" << EOF
server {
    listen 80;
    server_name $SERVER_NAME;
    client_max_body_size 200M;

    location / {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              \$host;
        proxy_set_header   X-Real-IP         \$remote_addr;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
}
EOF

ln -sf "$NGINXCONF" /etc/nginx/sites-enabled/photostudio
# remove default site if still present
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl reload nginx

# --- firewall -----------------------------------------------------------------
log "Firewall"
ufw allow 22/tcp  >/dev/null 2>&1 || true
ufw allow 80/tcp  >/dev/null 2>&1 || true
ufw allow 443/tcp >/dev/null 2>&1 || true
ufw --force enable >/dev/null 2>&1 || true
echo "  ports 22/80/443 open"

# --- TLS (only if domain + email provided) ------------------------------------
if [[ -n "$DEPLOY_DOMAIN" && -n "$DEPLOY_EMAIL" ]]; then
    # Check the domain resolves to this server before running certbot.
    SERVER_IP=$(curl -s https://api.ipify.org 2>/dev/null || echo "unknown")
    DOMAIN_IP=$(getent hosts "$DEPLOY_DOMAIN" | awk '{print $1; exit}' 2>/dev/null || echo "")
    if [[ "$SERVER_IP" == "$DOMAIN_IP" ]]; then
        log "TLS (Let's Encrypt)"
        certbot --nginx \
            -d "$DEPLOY_DOMAIN" \
            --non-interactive \
            --agree-tos \
            --no-eff-email \
            -m "$DEPLOY_EMAIL" \
            --redirect
        echo "  certificate issued and nginx updated"
    else
        echo ""
        echo "  ⚠  Skipping TLS: $DEPLOY_DOMAIN doesn't resolve to this server yet."
        echo "     Add a DNS A record pointing to $SERVER_IP, then re-run deploy.py."
    fi
fi

log "Setup complete"
# verify service is running
sleep 2
if systemctl is-active --quiet photostudio; then
    echo "  photostudio: running ✓"
else
    echo "  photostudio: NOT running — check logs: journalctl -u photostudio -n 50"
    exit 1
fi
