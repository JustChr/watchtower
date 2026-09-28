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
   and fill in the group ID, your user ID, the topic IDs and a model from
   `docker exec ollama ollama list`.
4. **Ollama network:** `docker network create --internal watchtower-llm`, and
   add `watchtower-llm` to the Ollama service's `networks:` (keeping `default`).

## Deploy

**Portainer:** Stacks → Add stack → *Repository* → this repo's URL (plus
credentials if it's private), reference `refs/heads/main`, compose path
`compose.yaml` → Deploy.

**CLI:** copy or clone this folder to the host, then `docker compose up -d --build`.

Both containers should turn *healthy* within a minute and post "online" into ⚙️ System.
Send `/status` or `/ping` in the group to check the gateway.

## Develop

```bash
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

Standard library only; Python 3.14.
