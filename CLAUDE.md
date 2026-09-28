# CLAUDE.md

Watchtower — watches GitHub repositories, has a **local** LLM (Ollama)
summarise new activity, and reports it to a Telegram group (one topic per kind
of work). Later phases draft triage replies and PR reviews, posted only after
approval in Telegram. First watched repo: `JustChr/BavarianData`.

Runs as a Docker/Portainer stack on a separate Ubuntu box. This Windows machine
is only for development: there is no Docker here, and the code can't be run
end-to-end locally.

## Layout

- `watchtower/` — the package. Standard library only; keep it that way unless a
  dependency clearly earns its place.
  - `watcher.py` — poll loop; holds the GitHub **read-only** token.
  - `gateway.py` — the only Telegram client (a bot allows one `getUpdates`
    poller); holds the bot token; answers only the configured user in the
    configured group.
  - `events.py` — GitHub lists → `Event`s (baseline, cursors; pure, tested).
  - `llm.py` — Ollama summaries; the model's answer is untrusted and parsed
    against a fixed schema.
  - `render.py` — Telegram HTML; every outside value is escaped.
  - `store.py` — shared SQLite (WAL): seen, cursors, outbox, heartbeats.
  - `history.py` — searchable copy (FTS5, own `history.db`) of every issue,
    PR, discussion and comment, plus the repo briefs; synced by the watcher.
  - `snapshot.py` — default-branch tarball → `/data/repos/<owner>/<name>`
    (safe unpack); gathers the repo's own docs and file tree.
  - `brief.py` — per new release (betas too), `agent_model` writes a repo
    brief from the docs at the tag + release notes (no releases: from main,
    at most weekly); used as
    prompt context only after the user approves it via a Telegram button.
  - Buttons: the gateway records presses as `decision` rows in the store;
    the watcher applies them. `callback_data` is `kind:action:id`.
  - `github.py`, `telegram.py`, `config.py`, `__main__.py`.
- `compose.yaml`, `Dockerfile` — one image, two services.
- `config.example.toml` — template; the real `config.toml` lives only on the host.
- `tests/` — pytest, no network.

## Commands

```
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

Python 3.14.

## Rules

- **Untrusted input everywhere.** Issue/PR/discussion text is written by
  strangers and goes into a model prompt. The model's output is untrusted too.
  Never let either reach a shell, a GitHub write, or unescaped Telegram HTML.
- **Least privilege per container.** A service holds only its own secret.
  Future worker agents get **no secrets and no internet** (only the internal
  `watchtower-llm` network to Ollama). Nothing gets the Docker socket.
- **Nothing posts to GitHub without the user's approval** in Telegram.
- **No `*-cloud` Ollama models** — prompts would leave the box (`config.py`
  refuses them).
- Personal data (chat/user/topic IDs, host names, addresses) stays in the
  host's `config.toml` or local memory, never in the repo.
- Every behavior change comes with a test in `tests/`.
