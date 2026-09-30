# Watchtower

Watches GitHub repositories, has a **local** LLM (Ollama) summarise new
activity, and reports it to a Telegram group, one topic per kind of work.

**Phase 1:** report. **Phase 2 (this):** draft replies to issues and
discussions, posted by a GitHub App only after you approve them in Telegram.
Later: PR reviews by a sandboxed worker agent.

## How it fits together

```
GitHub ─poll─> watcher ── jobs, drafts ──> worker ──watchtower-llm──> ollama
  ▲   (threads, files, code → /data)         │     (internal, no internet)
  │                                          ▼
  │                                  outbox (SQLite) ──> gateway ──> Telegram group
  │                                          ▲             │ presses,  ├ 🩺 Triage   issues + comments, drafts
  │                                          │             │ replies   ├ 🔍 Reviews  PRs + comments
  │                                          │             ▼           ├ 💬 Replies  discussions, drafts
  └───────────── approved text ──────────── poster <── decisions      └ ⚙️ System   startup, errors, /status
```

| Service | Holds | Networks |
|---|---|---|
| `watcher` | GitHub **read-only** token | `egress` only: no model |
| `gateway` | Telegram bot token | `egress` |
| `worker` | **nothing** | `watchtower-llm` only: no internet |
| `poster` | the GitHub App's private key | `egress` only: no model |
| `web` | **nothing** | the LAN: port 8780 on the box |

All model work (summaries, briefs, drafts, replays) is the **worker's**, one
job at a time: the only service that feeds strangers' text to a model holds no
secret and can't reach the internet. The watcher fetches what a job needs
(threads, attached files, code at a release) into `/data` first. Summaries go
first, and a long draft runs the waiting ones between its passes.

- Polling, not webhooks: no public endpoint. Unchanged repos answer `304`,
  which costs no rate limit.
- Only the gateway talks to Telegram (a bot allows one `getUpdates` poller).
  It answers only the configured user in the configured group.
- Everything from GitHub, and everything the model says, is untrusted: the
  model's answer must match a fixed JSON shape and loses links and
  `@mentions`; all text is HTML-escaped before it reaches Telegram.
- Bot activity (Dependabot, Actions) arrives silently and skips the model.
- The first start sets a baseline and reports only what's newer
  (`backfill_hours` to look back).
- Containers: non-root, read-only root filesystem, no capabilities, memory and
  PID limits, no Docker socket.

## What the agents learn a repo from

Every `history_minutes` the watcher refreshes, per repo:

| Source | Where | Trust |
|---|---|---|
| **History:** every issue, PR, discussion and comment, full-text searchable | `/data/history.db` | Strangers' text: data only. GitHub's `author_association` marks what maintainers wrote. |
| **Code snapshot:** the default branch, downloaded as a tarball when its head commit changes | `/data/repos/<owner>/<name>` | What the maintainers merged. Unpacked safely: size and file-count limits, nothing outside the folder, no links pointing out. |
| **Repo docs:** `CLAUDE.md`/`AGENTS.md`, README, CONTRIBUTING, `.claude/skills/*/SKILL.md`, `docs/**/*.md` | from the snapshot | The maintainers' own words. |
| **Your notes:** `docs/knowledge/*.md` in the watched repo, one topic per file: facts the code doesn't state (how the service behaves, its limits, known quirks) | the default branch's snapshot, always the newest | Yours: trusted, and ranked above the brief and the model's guesses. |
| **Repo brief:** what the project is, current stable and beta, architecture, what it supports, common problems, how issues are handled. `agent_model` writes it from the docs at the release's tag, its release notes, the recent releases and the file tree | `/data/history.db` | Model output: used **only after you approve it** in ⚙️ System (✅ Use it / 🗑 Discard). |

**Drafts** get your notes and the repo's docs whole, as far as a share of the
context allows (notes first; docs at the version the author runs, so a replay
can't read about a later fix); what doesn't fit, the model can search and read
while investigating. A wrong or weak draft that needed knowledge you have is
the cue for a new note.

Summaries get the approved brief as background. Until one is approved, they
get the README's opening. **Each new release gets a new brief**: betas too,
unless `brief_betas = false`. A repo without releases gets its brief from the
default branch instead, renewed at most weekly while the branch keeps moving.

Check it from the host:
```bash
sudo docker exec watchtower-watcher-1 python -m watchtower history JustChr/BavarianData
sudo docker exec watchtower-watcher-1 python -m watchtower history search JustChr/BavarianData soc stuck
sudo docker exec watchtower-watcher-1 python -m watchtower history show JustChr/BavarianData 42
ls /opt/watchtower/data/repos/JustChr/BavarianData
```

## Reply drafts

Every stranger's new issue or discussion gets a draft, and so does a
stranger's comment on one when the summary says it **needs a reply** (not
maintainers, bots or PRs). The worker asks for it; the watcher refreshes that
thread in the history, downloads its files and the code at the author's
version; then the **worker** has `agent_model` write it from:

- trusted, as instructions: the approved brief (else the README's opening),
  the repo's own `.claude/skills/triage/SKILL.md` and its issue forms;
- checked by code: which files are attached (a ticked "I have attached…" box
  proves nothing), and which of them the model gets;
- data: the thread with the message to answer marked; the attached **text
  files** (diagnostics JSON, logs), which the watcher downloads into
  `/data/attachments` without a token (up to 2 MB, text only, JSON minified, a
  long file keeps its start and end); and similar earlier threads from the
  history with the maintainers' answers (to spot duplicates, and to answer the
  way you do).

The draft's 📎 line shows each file as *read* or *not read*. The model is told
it can't see files it didn't get; if a reply still sounds as if it read one,
the draft carries a ⚠️ warning.

A draft takes several passes (big files read in parts, an investigation with
read-only tools, the 🧭 assessment, the reply). The assessment arrives as soon
as it's done; each finished pass is kept, so a restart resumes the draft
instead of starting over.

The draft arrives in the item's topic, in a copyable block, with
**✅ Post** / **🗑 Reject**:

- **Edit:** reply to the draft message with your version; it comes back as a
  new version with its own buttons. Telegram formatting (bold, code, links) is
  turned back into Markdown. Buttons on an older version no longer post.
- **Reject:** then reply to the draft with a reason, if you like.
- **Post:** the **poster** posts exactly that version as the GitHub App
  (`<app>[bot]`), and confirms with a link. If it's unclear whether GitHub got
  it (timeout, restart), it tells you to check instead of retrying.

The model's reply loses `@mentions` and every link outside the repo. If
several comments arrive in one thread, only the newest gets a draft, and a
newer draft outdates the older one. Every version, and every rejection reason,
stays in `watchtower.db` (`draft`, `draft_version`) as feedback for later.

**Replay** closed issues to measure the drafts (the report puts the model's
assessment and reply next to your real answer):
```bash
sudo docker exec watchtower-watcher-1 python -m watchtower eval JustChr/BavarianData 25 23 160@5
```
The watcher fetches the files and code, the worker replays; ⚙️ System says when
the report (`/opt/watchtower/data/eval/`) is done.

## Web UI

`http://<box>:8780` on your LAN: how Watchtower works (a live signal-box panel
of the services, lit where work is in transit), what it's doing now, and
everything it did: every message, job and draft, each draft's passes with the
investigation, the checked assessment and every version, and **every model
call with its full prompt and answer** (the worker records them in
`/data/trace.db`, kept 90 days). Your notes, the repo docs, briefs and releases
per repo.

No login, so it's built to be harmless without one: it holds no secret, and it
can only edit or reject a draft, or **send a version to Telegram**, where your
✅ Post tap is still what posts (the poster refuses a post from anywhere else).
Other websites can't use your browser against it: a POST needs a custom header
(which forces a CORS preflight it never grants), and it answers only to IP
addresses, `localhost` and the names in `[web] hosts` (DNS rebinding). Add the
name you open it by there:

```toml
[web]
hosts = ["jarvis.home.arpa"]
```

## Host setup (once)

1. **Secrets**, readable only by the container user (UID 10001):
   ```bash
   sudo mkdir -p /opt/watchtower/secrets
   echo -n "GitHub read token: "; read -rs T; echo; printf '%s' "$T" | sudo tee /opt/watchtower/secrets/github_read >/dev/null
   echo -n "Telegram bot token: "; read -rs T; echo; printf '%s' "$T" | sudo tee /opt/watchtower/secrets/telegram_token >/dev/null
   unset T
   sudo chown 10001:10001 /opt/watchtower/secrets/* && sudo chmod 400 /opt/watchtower/secrets/*
   ```
   GitHub token: fine-grained, only the watched repos, **read** on Issues,
   Pull requests, Discussions, Contents.
2. **Data directory:**
   ```bash
   sudo install -d -o 10001 -g 10001 -m 700 /opt/watchtower/data
   ```
3. **Config:** copy `config.example.toml` to `/opt/watchtower/config.toml`
   and fill in the group ID, your user ID, the topic IDs and the two models
   (`summary_model`, `agent_model`) from `docker exec ollama ollama list`.
4. **Ollama network:** `docker network create --internal watchtower-llm`, and
   add `watchtower-llm` to the Ollama service's `networks:` (keeping `default`).
5. **GitHub App** (posts approved replies): GitHub → Settings → Developer
   settings → GitHub Apps → New. Webhook off; repository permissions
   **Issues: Read and write**, **Discussions: Read and write**, **Pull
   requests: Read and write** (PR reviews; Metadata: read is added by itself); only on this account. Install it on the watched repos.
   Put its App ID into `github.app_id`, and a generated private key into
   `/opt/watchtower/secrets/github_app_key` (10001, mode 400 like the others).
   The stack won't start without that file.
6. **Sandbox for PR checks** (the `runner` service; the stack needs this even
   while `llm.run_checks = false`): it runs a stranger's code, so it needs
   gVisor (`runsc`) as a Docker runtime, and two folders it may use:
   ```bash
   sudo install -d -o 10001 -g 10001 -m 700 /opt/watchtower/sandbox /opt/watchtower/toolchain /opt/watchtower/tools
   ```
   (`toolchain` is the second service under gVisor: it has a network, but only for
   installing a repo's dependencies from its default branch into `tools`, which the
   runner reads. Turn the checks on with `llm.run_checks = true` once the selftest passes.)
   gVisor: install `runsc` from Google's apt repository (gvisor.dev/docs/user_guide/install),
   then `sudo runsc install && sudo systemctl restart docker`. Check with
   `sudo docker run --rm --runtime=runsc alpine dmesg | head -1` ("Starting gVisor").
   After deploying, prove the isolation: `sudo docker exec watchtower-watcher-1
   python -m watchtower selftest` must end in `ISOLATED`.

## Deploy

Every push to `main` runs the tests, then GitHub Actions builds the image and
publishes `ghcr.io/justchr/watchtower:latest`. The host only pulls it (Portainer's
bundled compose can't run `build:`).

**Portainer:** Stacks → Add stack → *Repository* → this repo's URL, reference
`refs/heads/main`, compose path `compose.yaml` → Deploy. With GitOps updates on,
a push redeploys by itself; otherwise *Pull and redeploy*.

**CLI:** copy or clone this folder to the host, then `docker compose up -d`.

The five containers with a healthcheck should turn *healthy* within a minute and post "online" into
⚙️ System (the poster names the App it posts as). Send `/status` or `/ping` in the
group to check the gateway.

## Develop

```bash
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

Standard library only, plus `cryptography` for the App's JWT and `PyYAML` to read a
repo's CI workflows; Python 3.14.
