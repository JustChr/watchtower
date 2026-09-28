"""The watcher process: poll GitHub, summarise, queue; keep each repo's history, code
snapshot and brief current. Holds the read-only GitHub token."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from . import attachments, brief, llm, render, snapshot
from .config import DATA_DIR, Config, read_secret
from .events import Event, Source, poll_repo
from .github import GitHub, GitHubError, RateLimited
from .history import MAINTAINERS, History, sync_repo, sync_thread
from .store import Store

_LOGGER = logging.getLogger(__name__)

# Report a repeating error to Telegram at most this often (seconds).
ERROR_REPORT_INTERVAL = 3600


def wants_draft(event: Event, summary: llm.Summary | None, cfg: Config) -> bool:
    """A stranger's issue, discussion or comment on one that the summary says needs a reply."""

    return (
        cfg.drafts
        and summary is not None
        and summary.needs_reply
        and event.thread_kind is not None
        and not event.is_bot
        and event.association not in MAINTAINERS
    )


def queue_draft(
    event: Event, source: Source, history: History, store: Store, files: Path | None = None
) -> bool:
    """Refresh the thread in the history and download its attachments into ``files``
    (the drafter can't reach GitHub), then queue a draft. Without a fresh thread
    there's no draft: it would answer an old state. Without the files there is one."""

    try:
        sync_thread(source, history, event.repo, event.number, event.thread_kind)
    except Exception as err:  # noqa: BLE001 -- the event is still reported
        _LOGGER.warning("%s: no draft, thread refresh failed: %s", event.key, err)
        return False
    if files is not None:
        try:
            attachments.download(history.thread(event.repo, event.number), event.repo, files)
        except Exception:  # noqa: BLE001 -- the draft says what it couldn't read
            _LOGGER.exception("%s: attachments", event.key)
    store.add_draft(
        event.key,
        repo=event.repo,
        number=event.number,
        kind=event.thread_kind,
        topic=event.topic,
        title=event.title,
        url=event.url,
        reply_to=event.reply_to,
    )
    return True


def handle(
    event: Event,
    cfg: Config,
    store: Store,
    context: str = "",
    source: Source | None = None,
    history: History | None = None,
    files: Path | None = None,
) -> None:
    # Bot activity (Dependabot, Actions) is reported silently and not summarised.
    summary = None if event.is_bot else llm.summarize(cfg, event.for_model(), context)
    drafting = (
        source is not None
        and history is not None
        and wants_draft(event, summary, cfg)
        and queue_draft(event, source, history, store, files)
    )
    store.enqueue(
        event.topic,
        render.message(event, summary, drafting),
        url=event.url,
        silent=event.is_bot,
        seen_key=event.key,
    )
    _LOGGER.info("queued %s%s", event.key, " (drafting)" if drafting else "")


def poll_once(
    source: Source,
    store: Store,
    cfg: Config,
    titles: dict[tuple[str, int], tuple[str, bool]],
    errors: dict[str, float],
    contexts: dict[str, str] | None = None,
    history: History | None = None,
    files: Path | None = None,
) -> float | None:
    """Poll every repo once. Returns the time to sleep until if rate limited.

    ``contexts`` maps a repo to background for its summaries (``brief.project_context``).
    Without ``history`` no drafts are queued; ``files`` is where their attachments go.
    """

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
                context = (contexts or {}).get(repo, "")
                handle(event, cfg, store, context, source, history, files)
        for name, value in poll.cursors.items():
            store.set_cursor(name, value)
    return None


def _failed(label: str, err: Exception, store: Store, errors: dict[str, float]) -> None:
    """Log a failed step; report it to Telegram at most once per ``ERROR_REPORT_INTERVAL``."""

    if isinstance(err, RateLimited):
        _LOGGER.warning("%s: %s", label, err)
        return
    if isinstance(err, GitHubError):
        _LOGGER.warning("%s: %s", label, err)
    else:
        _LOGGER.error("%s: unexpected error", label, exc_info=err)
    if time.time() - errors.get(label, 0) > ERROR_REPORT_INTERVAL:
        errors[label] = time.time()
        store.enqueue("system", render.system(f"{label}: {type(err).__name__}: {err}"))


def sync_history(
    source: Source, history: History, store: Store, cfg: Config, errors: dict[str, float]
) -> None:
    """Bring every repo's searchable history up to date; failures never stop the polling."""

    for repo in cfg.repos:
        was_empty = not any(history.counts(repo).values())
        try:
            written = sync_repo(source, history, repo)
        except Exception as err:  # noqa: BLE001
            _failed(f"History {repo}", err, store, errors)
            continue
        _LOGGER.info("history %s: %d record(s) written", repo, written)
        counts = history.counts(repo)
        if was_empty and any(counts.values()):
            store.enqueue(
                "system",
                render.system(
                    f"History of {repo} loaded: {counts['issue']} issues, {counts['pr']} PRs,"
                    f" {counts['discussion']} discussions, {counts['comment']} comments."
                ),
            )


def sync_code(
    source: snapshot.Source,
    history: History,
    store: Store,
    cfg: Config,
    errors: dict[str, float],
    root: Path,
) -> None:
    """Refresh each repo's code snapshot and, on a new release, put a new brief up for approval."""

    for repo in cfg.repos:
        try:
            commit = snapshot.sync(source, history, repo, root)
        except Exception as err:  # noqa: BLE001
            _failed(f"Snapshot {repo}", err, store, errors)
            continue
        if not cfg.agent_model:
            continue
        release_copy = snapshot.path_for(root / ".release", repo)
        try:
            releases = snapshot.releases(source, repo)
            history.put_releases(repo, releases)  # the drafter compares versions with them
            target = brief.plan(releases, commit, cfg.brief_betas)
            if not brief.due(history, repo, target, time.time()):
                continue
            store.beat("watcher", f"writing the brief for {repo}")  # can take minutes
            if target.release is None:
                text = brief.generate(cfg, repo, snapshot.path_for(root, repo))
            else:
                snapshot.fetch(source, repo, target.ref, release_copy)
                text = brief.generate(cfg, repo, release_copy, target.release, releases)
        except Exception as err:  # noqa: BLE001
            _failed(f"Brief {repo}", err, store, errors)
            continue
        finally:
            shutil.rmtree(release_copy, ignore_errors=True)
        brief_id = history.add_brief(repo, target.ref, target.label, text)
        if text is None:
            text = f"Brief for {repo} {target.label} failed; retrying in a day."
            store.enqueue("system", render.system(text))
            continue
        store.enqueue(
            "system",
            render.brief(repo, target.label, text),
            buttons=[
                ("✅ Use it", f"brief:approve:{brief_id}"),
                ("🗑 Discard", f"brief:reject:{brief_id}"),
            ],
        )


def apply_decisions(store: Store, history: History) -> None:
    for decision_id, action, ref, _ in store.open_decisions("brief"):
        history.decide_brief(ref, approved=action == "approve")
        store.mark_applied(decision_id)


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

    history = History(DATA_DIR / "history.db")
    root = DATA_DIR / "repos"
    titles: dict[tuple[str, int], tuple[str, bool]] = {}
    errors: dict[str, float] = {}
    synced = 0.0
    while True:
        started = time.time()
        apply_decisions(store, history)
        contexts = {repo: brief.project_context(history, repo, root) for repo in cfg.repos}
        resume_at = poll_once(
            github, store, cfg, titles, errors, contexts, history, DATA_DIR / "attachments"
        )
        history_due = cfg.history_minutes and started - synced >= cfg.history_minutes * 60
        if resume_at is None and history_due:
            synced = started
            sync_history(github, history, store, cfg, errors)
            sync_code(github, history, store, cfg, errors, root)
        store.beat("watcher", f"polled {len(cfg.repos)} repo(s)")
        if resume_at is not None:
            time.sleep(max(cfg.poll_seconds, resume_at - time.time() + 5))
        else:
            time.sleep(max(1, cfg.poll_seconds - (time.time() - started)))
