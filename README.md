# Watchtower

Watches GitHub repositories, has a **local** LLM (Ollama) summarise new
activity, and reports it to a Telegram group, one topic per kind of work.

**Phase 1 (this):** report only. Nothing can write to GitHub.
Later phases: drafted triage replies and PR reviews by a sandboxed worker
agent, posted only after approval in Telegram.

## How it fits together

```
GitHub ─poll─> watcher ──outbox (SQLite)──> gateway ──> Telegram group
                  │                                        ├ 🩺 Triage   issues + comments
                  └─ watchtower-llm ─> ollama               ├ 🔍 Reviews  PRs + comments
                     (internal network, no internet)       ├ 💬 Replies  discussions
                                                           └ ⚙️ System   startup, errors, /status
```

| Service | Holds | Networks |
|---|---|---|
| `watcher` | GitHub **read-only** token | `egress`, `watchtower-llm` |
| `gateway` | Telegram bot token | `egress` |

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
| **Repo brief:** what the project is, current stable and beta, architecture, what it supports, common problems, how issues are handled. `agent_model` writes it from the docs at the release's tag, its release notes, the recent releases and the file tree | `/data/history.db` | Model output: used **only after you approve it** in ⚙️ System (✅ Use it / 🗑 Discard). |

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

## Deploy

Every push to `main` runs the tests, then GitHub Actions builds the image and
publishes `ghcr.io/justchr/watchtower:latest`. The host only pulls it (Portainer's
bundled compose can't run `build:`).

**Portainer:** Stacks → Add stack → *Repository* → this repo's URL, reference
`refs/heads/main`, compose path `compose.yaml` → Deploy. With GitOps updates on,
a push redeploys by itself; otherwise *Pull and redeploy*.

**CLI:** copy or clone this folder to the host, then `docker compose up -d`.

Both containers should turn *healthy* within a minute and post "online" into ⚙️ System.
Send `/status` or `/ping` in the group to check the gateway.

## Develop

```bash
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

Standard library only; Python 3.14.
