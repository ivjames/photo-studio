FROM python:3.12-slim

WORKDIR /opt/photostudio

# Runtime deps for opencv-python-headless, plus Tailscale (to reach a
# tailnet-only Ollama server from App Platform). python:3.12-slim is bookworm.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 curl gnupg ca-certificates \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.noarmor.gpg \
        -o /usr/share/keyrings/tailscale-archive-keyring.gpg \
    && curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.tailscale-keyring.list \
        -o /etc/apt/sources.list.d/tailscale.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends tailscale \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY photo_studio.py scan_splitter.py photo_tagger.py ./
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Default data dir — override via PHOTOSTUDIO_HOME; mount a volume here in prod
RUN mkdir -p /data
ENV PHOTOSTUDIO_HOME=/data

# DO App Platform injects PORT; default to 8080 for local docker run
ENV PORT=8080

# Entrypoint brings up Tailscale (if TS_AUTHKEY is set) then execs gunicorn.
ENTRYPOINT ["/docker-entrypoint.sh"]
