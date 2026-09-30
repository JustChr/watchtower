"""The watcher process: poll GitHub and hand what's new to the worker; keep each
repo's history and code snapshot current; fetch what a draft or a brief needs.
Holds the read-only GitHub token and never talks to the model: all model work is
the worker's, which has no secrets and no internet."""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from pathlib import Path

from . import attachments, brief, checking, drafts, pr, render, snapshot
from .config import DATA_DIR, Config, read_secret
from .events import Event, Source, poll_repo
from .github import GitHub, GitHubError, RateLimited
from .history import History, sync_repo, sync_thread
from .store import Draft, Store

_LOGGER = logging.getLogger(__name__)

# Report a repeating error to Telegram at most this often (seconds).
ERROR_REPORT_INTERVAL = 3600
# The volume shared with the runner (see ``sandbox``).
SANDBOX_DIR = Path(os.environ.get("WATCHTOWER_SANDBOX", "/sandbox"))
TOOLCHAIN_DIR = Path(os.environ.get("WATCHTOWER_TOOLCHAIN", "/toolchain"))
TOOLS_DIR = Path(os.environ.get("WATCHTOWER_TOOLS", "/tools"))
# Polls a draft's thread refresh may fail before the draft is given up.
PREP_TRIES = 3


def handle(event: Event, cfg: Config, store: Store) -> None:
    """Bot activity (Dependabot, Actions) is reported at once, silently, without a
    summary; everything else goes to the worker, which summarises and reports it."""

    if event.is_bot:
        store.enqueue(
            event.topic,
            render.message(event, None),
            url=event.url,
            silent=True,
            seen_key=event.key,
        )
    else:
        store.add_job("summary", event.key, dataclasses.asdict(event), seen_key=event.key)
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


def prepare(
    draft: Draft, source: Source, history: History, files: Path | None, root: Path | None
) -> bool:
    """Refresh the draft's thread in the history, download its attachments into
    ``files`` and the code at the author's version under ``root``: the worker can't
    reach GitHub. Without a fresh thread it isn't ready (it would answer an old
    state); without the files or the code it is."""

    label = (draft.repo, draft.number)
    try:
        sync_thread(source, history, draft.repo, draft.number, draft.kind)
    except Exception as err:  # noqa: BLE001 -- tried again next poll
        _LOGGER.warning("%s#%d: thread refresh failed: %s", *label, err)
        return False
    thread = history.thread(draft.repo, draft.number)
    if draft.kind == "pr":  # a review reads the diff and the code at the PR's head
        if root is not None:
            try:
                pr.fetch(source, draft.repo, draft.number, root)
            except Exception as err:  # noqa: BLE001 -- tried again next poll
                _LOGGER.warning("%s#%d: pull request fetch failed: %s", *label, err)
                return False
        return True
    if files is not None:
        try:
            attachments.download(thread, draft.repo, files)
        except Exception:  # noqa: BLE001 -- the draft says what it couldn't read
            _LOGGER.exception("%s#%d: attachments", *label)
    if root is not None:
        try:
            drafts.fetch_code(source, history, draft.repo, thread, files, root)
        except Exception as err:  # noqa: BLE001 -- the draft uses the default branch
            _LOGGER.warning("%s#%d: code at the author's version: %s", *label, err)
    return True


def queue_review(source: Source, store: Store, repo: str, number: int) -> str:
    """Ask for a review of an existing PR, whoever wrote it (``python -m watchtower review``):
    a draft in ``prep`` like a new stranger's PR gets. Returns what to tell the user."""

    item = source.get_json(f"/repos/{repo}/issues/{number}")
    if "pull_request" not in item:
        return f"{repo}#{number} is not a pull request"
    key = f"{repo}#review-{number}-{int(time.time())}"  # a new one each time it is asked for
    store.add_draft(
        key,
        repo=repo,
        number=number,
        kind="pr",
        topic="reviews",
        title=item["title"],
        url=item["html_url"],
        status="prep",
    )
    return f"queued a review of {repo}#{number}: {item['title']}"


def _settled(draft, source, store, root, box, toolbox, tools) -> bool:
    """Whether the draft's checks have settled; an error is logged and tried again next
    poll (it must not stop the watcher)."""

    try:
        return checking.settle(draft, source, store, root, box, toolbox=toolbox, tools=tools)
    except Exception:  # noqa: BLE001
        _LOGGER.exception("%s#%d: checks", draft.repo, draft.number)
        return False


def prepare_drafts(
    source: Source,
    history: History,
    store: Store,
    failures: dict[int, int],
    files: Path | None = None,
    root: Path | None = None,
    box: Path | None = None,
    toolbox: Path | None = None,
    tools: Path | None = None,
) -> None:
    """Hand the worker the drafts it asked for, once their thread is fetched. One
    whose thread can't be refreshed ``PREP_TRIES`` polls in a row fails, visibly.

    ``box``: the sandbox volume. A PR's review then also waits for its checks to settle
    (``checking``): they take minutes, and the draft stays in ``prep`` meanwhile."""

    for draft in store.drafts("prep"):
        checked = draft.kind == "pr" and box is not None and root is not None
        current = checking.state(store, draft) if checked else None
        if current and current["state"] in checking.OPEN:  # thread and PR already fetched
            if _settled(draft, source, store, root, box, toolbox, tools):
                store.prepared(draft.id)
            continue
        if prepare(draft, source, history, files, root):
            failures.pop(draft.id, None)
            if checked and not checking.settle(
                draft, source, store, root, box, toolbox=toolbox, tools=tools
            ):
                continue
            store.prepared(draft.id)
            continue
        failures[draft.id] = failures.get(draft.id, 0) + 1
        if failures[draft.id] >= PREP_TRIES:
            failures.pop(draft.id)
            store.fail_draft(draft.id)
            text = (
                f"The draft reply to {draft.repo}#{draft.number} failed:"
                " the thread could not be fetched from GitHub."
            )
            store.enqueue(draft.topic, render.system(text), url=draft.url)


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
    """Refresh each repo's code snapshot and, on a new release, fetch the code at its
    tag and queue a brief for the worker to write."""

    for repo in cfg.repos:
        try:
            commit = snapshot.sync(source, history, repo, root)
        except Exception as err:  # noqa: BLE001
            _failed(f"Snapshot {repo}", err, store, errors)
            continue
        if not cfg.agent_model:
            continue
        try:
            releases = snapshot.releases(source, repo)
            history.put_releases(repo, releases)  # the worker compares versions with them
            target = brief.plan(releases, commit, cfg.brief_betas)
            if not brief.due(history, repo, target, time.time()):
                continue
            tag = target.release.tag if target.release else None
            if tag is not None and snapshot.fetch_version(source, repo, tag, root) is None:
                raise snapshot.SnapshotError(f"release tag {tag!r} can't be a folder name")
        except Exception as err:  # noqa: BLE001
            _failed(f"Brief {repo}", err, store, errors)
            continue
        payload = {"repo": repo, "ref": target.ref, "label": target.label, "tag": tag}
        store.add_job("brief", f"brief:{repo}:{target.ref}", payload)


def apply_decisions(store: Store, history: History) -> None:
    for decision_id, action, ref, _ in store.open_decisions("brief"):
        history.decide_brief(ref, approved=action == "approve")
        store.mark_applied(decision_id)


def run(cfg: Config) -> None:
    store = Store(DATA_DIR / "watchtower.db")
    github = GitHub(read_secret("github_read"))
    _LOGGER.info("watcher started")
    store.enqueue("system", render.system(f"Watcher online: {', '.join(cfg.repos)}."))

    history = History(DATA_DIR / "history.db")
    root = DATA_DIR / "repos"
    titles: dict[tuple[str, int], tuple[str, bool]] = {}
    errors: dict[str, float] = {}
    failures: dict[int, int] = {}
    synced = 0.0
    while True:
        started = time.time()
        apply_decisions(store, history)
        resume_at = poll_once(github, store, cfg, titles, errors)
        if resume_at is None:
            prepare_drafts(
                github,
                history,
                store,
                failures,
                DATA_DIR / "attachments",
                root,
                SANDBOX_DIR if cfg.run_checks and cfg.reviews else None,
                TOOLCHAIN_DIR if TOOLCHAIN_DIR.is_dir() else None,
                TOOLS_DIR if TOOLS_DIR.is_dir() else None,
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
