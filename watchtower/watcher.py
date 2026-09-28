"""The watcher process: poll GitHub, summarise, queue. Holds the read-only GitHub token."""

from __future__ import annotations

import logging
import time

from . import llm, render
from .config import DATA_DIR, Config, read_secret
from .events import Event, Source, poll_repo
from .github import GitHub, GitHubError, RateLimited
from .store import Store

_LOGGER = logging.getLogger(__name__)

# Report a repeating error to Telegram at most this often (seconds).
ERROR_REPORT_INTERVAL = 3600


def handle(event: Event, cfg: Config, store: Store) -> None:
    # Bot activity (Dependabot, Actions) is reported silently and not summarised.
    summary = None if event.is_bot else llm.summarize(cfg, event.for_model())
    store.enqueue(
        event.topic,
        render.message(event, summary),
        url=event.url,
        silent=event.is_bot,
        seen_key=event.key,
    )
    _LOGGER.info("queued %s", event.key)


def poll_once(
    source: Source,
    store: Store,
    cfg: Config,
    titles: dict[tuple[str, int], tuple[str, bool]],
    errors: dict[str, float],
) -> float | None:
    """Poll every repo once. Returns the time to sleep until if rate limited."""

    for repo in cfg.repos:
        try:
            poll = poll_repo(source, store, cfg, repo, titles)
        except RateLimited as err:
            _LOGGER.warning("%s: %s", repo, err)
            return err.reset_at
        except Exception as err:  # noqa: BLE001 -- one bad repo mustn't stop the others
            if isinstance(err, GitHubError):
                _LOGGER.warning("%s: %s", repo, err)
            else:
                _LOGGER.exception("%s: unexpected error", repo)
            if time.time() - errors.get(repo, 0) > ERROR_REPORT_INTERVAL:
                errors[repo] = time.time()
                store.enqueue("system", render.system(f"{repo}: {type(err).__name__}: {err}"))
            continue
        errors.pop(repo, None)
        for event in poll.events:
            if not store.is_seen(event.key):
                handle(event, cfg, store)
        for name, value in poll.cursors.items():
            store.set_cursor(name, value)
    return None


def _check_model(cfg: Config) -> str:
    if not cfg.summary_model:
        return "summaries off (no llm.summary_model)"
    try:
        models = llm.available_models(cfg)
    except Exception as err:  # noqa: BLE001
        return (
            f"Ollama unreachable at {cfg.llm_url} ({type(err).__name__}); sending without summaries"
        )
    if not llm.has_model(models, cfg.summary_model):
        return f"model {cfg.summary_model} not in `ollama list`; sending without summaries"
    return f"summaries by {cfg.summary_model}"


def run(cfg: Config) -> None:
    store = Store(DATA_DIR / "watchtower.db")
    github = GitHub(read_secret("github_read"))
    status = _check_model(cfg)
    _LOGGER.info("watcher started: %s", status)
    store.enqueue("system", render.system(f"Watcher online: {', '.join(cfg.repos)}; {status}."))

    titles: dict[tuple[str, int], tuple[str, bool]] = {}
    errors: dict[str, float] = {}
    while True:
        started = time.time()
        resume_at = poll_once(github, store, cfg, titles, errors)
        store.beat("watcher", f"polled {len(cfg.repos)} repo(s)")
        if resume_at is not None:
            time.sleep(max(cfg.poll_seconds, resume_at - time.time() + 5))
        else:
            time.sleep(max(1, cfg.poll_seconds - (time.time() - started)))
