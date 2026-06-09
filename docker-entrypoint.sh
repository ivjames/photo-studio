#!/bin/sh
# Brings the container onto the tailnet (userspace mode, no TUN/privileges needed)
# so the app can reach a Tailscale-only Ollama server, then launches gunicorn.
#
# Requires TS_AUTHKEY (a reusable, ephemeral, tagged auth key). State is kept in
# memory (--state=mem:) because App Platform's filesystem is ephemeral; each
# deploy registers a fresh ephemeral node that auto-removes when it goes offline.
#
# tailscaled exposes a combined SOCKS5 + HTTP outbound proxy on localhost:1055.
# The app's Ollama call uses urllib's default opener, which honors http_proxy,
# so OLLAMA_URL traffic is tunneled into the tailnet without code changes.
set -e

if [ -n "${TS_AUTHKEY}" ]; then
  echo "[entrypoint] starting tailscaled (userspace networking)..."
  tailscaled \
    --tun=userspace-networking \
    --state=mem: \
    --socks5-server=localhost:1055 \
    --outbound-http-proxy-listen=localhost:1055 &

  # Wait for the daemon socket before `up`.
  i=0
  while [ $i -lt 20 ]; do
    tailscale status >/dev/null 2>&1 && break
    i=$((i + 1)); sleep 0.5
  done

  echo "[entrypoint] tailscale up (hostname=${TS_HOSTNAME:-photo-studio})..."
  tailscale up \
    --authkey="${TS_AUTHKEY}" \
    --hostname="${TS_HOSTNAME:-photo-studio}" \
    ${TS_EXTRA_ARGS:-}
  tailscale status || true
else
  echo "[entrypoint] TS_AUTHKEY unset — skipping Tailscale; a tailnet-only Ollama will be unreachable."
fi

exec gunicorn -w 1 --threads 8 --timeout 600 -b "0.0.0.0:${PORT}" photo_studio:app
