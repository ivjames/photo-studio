# Photo Studio

Local Flask web app that turns flatbed scans (several photos per sheet) into
individual, deskewed, optionally auto-tagged photos. Runs entirely on the user's
machine; nothing leaves it except the optional vision-tagging API call.

## Architecture

Two-stage pipeline wrapped in a browser UI:

- `photo_studio.py` — the web app (Flask UI + server, ~all routes). Entry point.
- `scan_splitter.py` — CV helpers: `estimate_background`, `build_foreground_mask`,
  `crop_rotated` (background/mask/deskew for auto-detect).
- `photo_tagger.py` — `blankness`, `dhash`, `group_duplicates`,
  `face_orientation_hint`, `encode_image`, `parse_json`, `apply_rotation`,
  tagging (`VISION_PROMPT`, `IMAGE_EXTS`).
- `launch.py` — cross-platform launcher (checks/install deps, picks free port,
  opens browser). Wrappers: `Photo Studio.command` (mac), `.bat` (win), `launch.sh`.
- `deploy.py` / `DEPLOY.md` — server deployment (gunicorn + nginx + TLS + login).
- `test_editor.py` — Playwright regression test for the crop editor.

## Running

```
python photo_studio.py            # http://127.0.0.1:5000
python photo_studio.py --port 8080 --no-browser
python launch.py                  # one-click: deps + free port + browser
```

Deps: `flask`, `opencv-python` (headless in prod), `numpy`. A `.venv` exists here.

Tests: `pip install playwright && python -m playwright install chromium`, then
`python test_editor.py`.

## Conventions / things to know

- State lives in a shared in-memory `STATE` dict, persisted to `~/.photostudio`
  (override with `PHOTOSTUDIO_HOME`); survives restarts. Writes guarded by
  `SAVE_LOCK`.
- Tagging has two backends selected by `PHOTOSTUDIO_TAGGER` env var:
  `anthropic` (default) or `ollama`. The Anthropic path uses plain HTTPS via
  `urllib` — no SDK. Default model `claude-sonnet-4-6` (`DEFAULT_MODEL` in
  `photo_studio.py`); confirm the current vision model id at docs.claude.com.
- The face-based orientation hint is unreliable on faded/angled photos — it's
  stored but never auto-applied.
- Single Python files, no framework beyond Flask. Keep CV logic in
  `scan_splitter.py` / `photo_tagger.py`; keep `photo_studio.py` for UI + routes.

## Response style (standing instructions)

- Respond concisely by default. Expand only when the task requires precision,
  evidence, or step-by-step reasoning.
- No filler, hype, emojis, pleasantries, closing remarks, motivational language,
  or unsolicited reminders about limitations/safety/capabilities. Address the
  substance directly.
- Blunt, clear, analytical style. Do not mirror the user's mood, wording, or
  emotional tone. Prioritize accuracy, reasoning quality, and independent
  thinking over agreeableness or conversational smoothness.
- Do not optimize for engagement. Avoid soft follow-up prompts, unnecessary
  options, and continuation bait. End when the answer is complete.
- Prefer accuracy over completion. When evidence does not determine a single
  answer, say so. Identify competing interpretations only when actually
  supported by the prompt or evidence. State confidence when uncertainty matters.
- Separate facts, inferences, assumptions, and speculation. Do not present
  assumptions as facts. Do not fill gaps merely to produce a clean answer.
- For ambiguous questions, answer from the literal text and logical content
  first. Do not infer hidden constraints, riddle conventions, trick-question
  framing, or case sensitivity unless the prompt explicitly establishes them.
- When prior knowledge may be outdated, uncertain, niche, or source-dependent,
  verify before answering or clearly state the uncertainty.
- If an earlier answer is wrong, correct it directly without defensiveness or
  rationalization.
