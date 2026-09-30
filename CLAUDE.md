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
  (RS256 for the GitHub App JWT) and `PyYAML` (`gates.py`: CI workflows); keep
  it that way unless a dependency clearly earns its place.
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
  - `handoff.py` — a bug (`our_bug`) is handed to the maintainer, not fixed:
    code decides confirmed vs suspected (high confidence, all quotes checked,
    an existing `file:line`, nothing missing), builds a prompt for Claude Code
    (sent after the draft; copy button in the web UI) with the findings
    framed as data; a confirmed issue gets the `bug` label on ✅ (the
    "Post only" button leaves it off).
  - `pr.py` + `reviews.py` — a stranger's new PR gets a **review draft** (a
    `draft` of kind `pr`, same staged pipeline, Telegram approval, revise and
    web UI). The watcher (`pr.fetch`) puts the PR's facts, per-file patches and
    the code at its head commit under `/data/repos/.prs/<owner>/<name>/<n>/`;
    the worker: facts computed by code (tests, docs, changelog, gate/CI/
    dependency files, conflicts, linked issue) → investigation (`investigate`'s
    reviewing mode: `list_changes`/`read_patch` + code tools) → assessment
    (findings quote a patch/code line, code checks the quote and finds the
    line itself) → review text (code appends the reviewed commit). The
    project's rules come from the **default branch**, never from the PR's own
    docs. Posted by the poster as one review with event `COMMENT` (never
    approve/merge); this token asks for `pull_requests: write` only then.
    Nothing here runs the PR's code (planned: a no-network `runner` service).
  - `gates.py` — what "the checks" are for a repo, read from its own default
    branch: `.github/workflows` `run:` steps of pull_request jobs (install
    commands = setup, the rest = gates; third-party actions, secrets, network
    or deploy steps are listed as not run), else conventions (pyproject,
    package.json, Makefile); `.watchtower/gates.toml` overrides. Plans only;
    runs nothing.
  - `sandbox.py` — the runner protocol: the watcher writes a job (merged
    tree + steps) under `/sandbox/jobs/<id>/`, the runner (no network, no
    secrets, gVisor, one job per container, never `/data`) executes it with a
    clean environment, per-step and total clocks, capped output, and writes
    `result.json` only after sweeping every process the steps started. The
    result is **untrusted**: `parse_result` accepts one fixed shape, sizes capped.
    `toolchain.py` (its own service: network, no secrets, no `/data`) installs the
    default branch's `setup` steps into a per-repo venv + `node_modules` under
    `/tools` (key = hash of dependency files; swapped in only when all steps
    passed); the runner reads it read-only. `runner.py` is the container's loop (one job, then exit → compose restarts
    it fresh); `checking.py` is the watcher's part: a review draft stays in
    `prep` while its `gates` stage settles (`none`/`conflict`/`unknown`/
    `waiting`/`running`/`done`/`timeout`), one job in the sandbox at a time;
    `reviews.gate_report` writes the "Checks" section by code. Off unless
    `llm.run_checks`. `python -m watchtower selftest` (in the watcher container)
    proves the isolation on the box.
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
    outbox `ref` is `draft:<id>` (instructions, own text after `text:`,
    reject reasons) — as `decision` rows; an instruction becomes a `revise`
    job: the worker has the model rework the latest version (`drafts.revise`);
    the watcher (briefs) or the poster (drafts) applies them. `callback_data`
    is `kind:action:id`.
  - `github.py`, `telegram.py`, `config.py`, `__main__.py`.
- `compose.yaml`, `Dockerfile` — one image, seven services (the `runner`
  runs under gVisor with no network, no secrets, no `/data`; the `toolchain`
  has a network of its own and nothing else).
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
