#!/usr/bin/env python3
"""
deploy.py — deploy Photo Studio to a remote Ubuntu server.

Copies the app files and runs setup.sh on the server over SSH.
Works on Mac, Linux, or Windows (WSL/Git Bash). No extra Python packages needed.

Usage:
    python deploy.py --host root@1.2.3.4 \\
        --domain photos.example.com \\
        --email you@example.com \\
        --users "alice:pw1,bob:pw2" \\
        --ollama-url "http://100.x.y.z:11434" \\
        [--model gemma4] [--tagger ollama] [--key ~/.ssh/id_rsa]

Re-run the same command to push updated app code (setup.sh is idempotent).

Tip: skip --domain for a quick first test over plain HTTP. Add it later once
your DNS A record points to the droplet.

Note: password values should not contain double-quotes.
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

APP_FILES = ["photo_studio.py", "scan_splitter.py", "photo_tagger.py"]
HERE = Path(__file__).parent.resolve()


def run(cmd, **kw):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        sys.exit(f"\nCommand failed (exit {result.returncode}).")
    return result


def ssh_opts(key):
    opts = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    if key:
        opts += ["-i", str(Path(key).expanduser())]
    return opts


def main():
    ap = argparse.ArgumentParser(description="Deploy Photo Studio to a remote server.")
    ap.add_argument("--host", required=True, help="SSH target, e.g. root@1.2.3.4")
    ap.add_argument("--domain", default="",
                    help="Domain name (e.g. photos.example.com). Skip for HTTP-only.")
    ap.add_argument("--email", default="",
                    help="Email for Let's Encrypt (required when --domain is set).")
    ap.add_argument("--users", default="admin:changeme",
                    help="Comma-separated user:password pairs, e.g. 'alice:pw1,bob:pw2'.")
    ap.add_argument("--tagger", default="ollama", choices=["ollama", "anthropic"])
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434",
                    help="Ollama endpoint (Tailscale IP, e.g. http://100.x.y.z:11434).")
    ap.add_argument("--model", default="gemma4")
    ap.add_argument("--key", default="", help="Path to SSH private key.")
    ap.add_argument("--anthropic-key", default="",
                    help="Anthropic API key (only for --tagger anthropic).")
    args = ap.parse_args()

    if args.domain and not args.email:
        sys.exit("--email is required when --domain is set (needed for Let's Encrypt).")

    opts = ssh_opts(args.key)

    # Verify local files exist.
    missing = [f for f in APP_FILES + ["setup.sh"] if not (HERE / f).exists()]
    if missing:
        sys.exit(f"Missing files: {', '.join(missing)}\nRun from the photostudio folder.")

    # Write environment config to a temp file — scp'd to the server and sourced
    # by setup.sh.  systemd EnvironmentFile format: KEY=VALUE (double-quoted).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env",
                                    delete=False, dir="/tmp") as f:
        env_path = f.name
        env_vars = {
            "PHOTOSTUDIO_TAGGER": args.tagger,
            "OLLAMA_URL": args.ollama_url,
            "OLLAMA_MODEL": args.model,
            "PHOTOSTUDIO_USERS": args.users,
            "PHOTOSTUDIO_HOME": "/opt/photostudio/data",
            "ANTHROPIC_API_KEY": args.anthropic_key,
        }
        for key, val in env_vars.items():
            # Escape any " in the value so the systemd unit file stays valid.
            safe = val.replace('"', '\\"')
            f.write(f'{key}="{safe}"\n')
        f.write(f'DEPLOY_DOMAIN="{args.domain}"\n')
        f.write(f'DEPLOY_EMAIL="{args.email}"\n')

    try:
        print("\n── Copying files ──────────────────────────────────────────────")
        files_to_copy = [str(HERE / f) for f in APP_FILES + ["setup.sh"]] + [env_path]
        run(["scp"] + opts + files_to_copy + [f"{args.host}:/tmp/"])

        print("\n── Running setup on server ────────────────────────────────────")
        env_remote = f"/tmp/{os.path.basename(env_path)}"
        run(["ssh"] + opts + [args.host, "bash", "/tmp/setup.sh", env_remote])

    finally:
        os.unlink(env_path)

    host_ip = args.host.split("@")[-1]
    url = f"https://{args.domain}" if args.domain else f"http://{host_ip}"

    print("\n──────────────────────────────────────────────────────────────────")
    print(f"  Photo Studio is live at {url}")
    print(f"  Accounts: {args.users}")
    print()
    if not args.domain:
        print("  ⚠  Running over plain HTTP. Point a domain at this IP and re-run")
        print("     with --domain and --email to add HTTPS.")
    if args.ollama_url.startswith("http://127.0.0.1"):
        print("  ⚠  Ollama URL is localhost — tagging won't work from the server.")
        print("     To connect your home Ollama via Tailscale:")
        print("       1. On droplet:       sudo tailscale up  (auth in browser)")
        print("       2. On home machine:  tailscale ip -4    (get tailnet IP)")
        print("       3. Re-run:           python deploy.py ... --ollama-url http://<ip>:11434")
    print()
    print(f"  Logs:    ssh {args.host} 'journalctl -u photostudio -f'")
    print(f"  Restart: ssh {args.host} 'systemctl restart photostudio'")
    print("──────────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
