# CLAUDE.md

Watchtower — watches GitHub repositories, has a **local** LLM (Ollama)
summarise new activity, and reports it to a Telegram group (one topic per kind
of work). Phase 2 drafts replies to issues and discussions, posted by a GitHub
App only after approval in Telegram; phase 3 adds PR reviews. First watched
repo: `JustChr/BavarianData`.

Runs as a Docker/Portainer stack on a separate Ubuntu box. This Windows machine
is only for development: there is no Docker here, and the code can't be run
end-to-end locally.

## Layout

- `watchtower/` — the package. Standard library only, except `cryptography`
  (RS256 for the GitHub App JWT); keep it that way unless a dependency clearly
  earns its place.
  - `watcher.py` — poll loop; holds the GitHub **read-only** token; never
    calls the model: queues `job`s (summary, brief, eval) for the worker, and
    fetches what they need (a draft's thread, attachments, code at a release).
  - `worker.py` — **all** model work, one job at a time (summaries first, run
    between a long job's passes too; then drafts; then briefs, replays);
    **no secrets, no internet** (only `watchtower-llm`); decides which events
    get a draft. Draft logic in `drafts.py`: passes (files, investigation,
    assessment, reply), each finished one kept as a `draft_stage`, so a
    restart resumes; the assessment goes to Telegram when it's done; attached text
    files come from `attachments.py` (the watcher downloads them, tokenless).
    Before the assessment, `investigate.py` lets the model look things up with
    read-only tools (code at the author's version, which the watcher fetches
    into `/data/repos/.versions/`; attached files; history), bounded by
    `agent_steps`; tool arguments are untrusted, paths confined to the copy.
    Repo knowledge in drafts: the maintainer's notes (`docs/knowledge/*.md`
    in the watched repo, from the default branch) and the repo's docs (at
    the judged version), whole within `DOCS_SHARE`, the rest via `search_docs`.
  - `poster.py` — the only GitHub writer: holds the App key (`github_app.py`),
    applies draft decisions, posts exactly the approved version; no model.
  - `web.py` — the web UI (stdlib `http.server`, JSON API) + `web/` (one
    vanilla JS/CSS page, vendored Barlow fonts; no build step). LAN, no
    login, holds no secret: it can edit/reject drafts and "Post" = re-offer in
    Telegram; decisions carry `origin` (`telegram`/`web`) and the poster posts
    only Telegram ones. Guards: `X-Watchtower` header on POST, Host allowlist
    (`[web] hosts`), strict CSP; outside text is built as DOM text, never HTML.
  - `trace.py` — every model call (full prompt + answer, subject, step,
    seconds) in `/data/trace.db`, recorded by the worker via `llm.tracer`,
    pruned after 90 days.
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
  - `evaluate.py` — replays closed issues (`eval` CLI in the watcher: it
    prepares, the worker replays).
  - Buttons: the gateway records presses — and replies to messages whose
    outbox `ref` is `draft:<id>` (edits, reject reasons) — as `decision` rows;
    the watcher (briefs) or the poster (drafts) applies them. `callback_data`
    is `kind:action:id`.
  - `github.py`, `telegram.py`, `config.py`, `__main__.py`.
- `compose.yaml`, `Dockerfile` — one image, five services.
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
