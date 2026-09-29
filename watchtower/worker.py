"""The worker process: all model work -- summaries, briefs, reply drafts, replays.

It is the one process that feeds strangers' text to a model, so it holds no
secret and has no internet: its only network is ``watchtower-llm`` to Ollama.
The watcher hands it jobs and fetches what they need (threads, attachments, code
at a release); the worker reads those from the shared volume and writes its
results to the store: messages for the gateway, drafts for the poster.

One job at a time, and one model call at a time: the big model's context is
what's scarce on the box. Summaries come first -- a one-line notification
shouldn't wait for a draft -- and a long job runs the waiting summaries between
its passes. Then drafts, then briefs and replays.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from . import brief, drafts, evaluate, llm, render, snapshot
from .config import DATA_DIR, Config
from .events import Event
from .history import MAINTAINERS, History
from .store import Draft, Job, Store

_LOGGER = logging.getLogger(__name__)

IDLE_SECONDS = 5

# New threads: every one from a stranger gets a draft, whatever the summary says.
OPENING_KINDS = frozenset({"issue", "discussion"})


class Worker:
    def __init__(
        self, cfg: Config, store: Store, history: History, root: Path, files: Path | None = None
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.history = history
        self.root = root  # the code snapshots
        self.files = files  # the downloaded attachments
        self.busy = ""  # the long job in progress, for the heartbeat

    # -- the loop -------------------------------------------------------------

    def between(self, step: str) -> None:
        """Called between the passes of a long job: a heartbeat (each pass can take up
        to agent_timeout), then the summaries that came in meanwhile."""

        self.store.beat("worker", f"{self.busy}: {step}" if self.busy else step)
        self.summaries()

    def summaries(self) -> int:
        done = 0
        while (job := self.store.claim_job(("summary",))) is not None:
            self._run(job, self.summary)
            done += 1
        return done

    def step(self) -> bool:
        """Do the next piece of work. Returns whether there was any."""

        if self.summaries():
            return True
        draft = self.store.claim_draft() if self.cfg.drafts else None
        if draft is not None:
            try:
                self.draft(draft)
            except Exception:  # noqa: BLE001 -- one bad draft mustn't stop the worker
                _LOGGER.exception("draft %d: unexpected error", draft.id)
                self.store.fail_draft(draft.id)
            return True
        job = self.store.claim_job(("brief", "eval"))
        if job is not None:
            self.busy = job.key
            try:
                self._run(job, self.brief if job.kind == "brief" else self.replay)
            finally:
                self.busy = ""
            return True
        return False

    def _run(self, job: Job, work) -> None:
        try:
            work(job)
        except Exception:  # noqa: BLE001 -- one bad job mustn't stop the worker
            _LOGGER.exception("%s job %d: unexpected error", job.kind, job.id)
            self.store.finish_job(job.id, failed=True)

    # -- summaries --------------------------------------------------------------

    def summary(self, job: Job) -> None:
        """Summarise an event and report it; ask the watcher to prepare a draft if it
        wants one."""

        cfg, store = self.cfg, self.store
        event = Event(**job.payload)
        context = brief.project_context(self.history, event.repo, self.root)
        summary = llm.summarize(cfg, event.for_model(), context)
        drafting = wants_draft(event, summary, cfg)
        if drafting:
            store.add_draft(
                event.key,
                repo=event.repo,
                number=event.number,
                kind=event.thread_kind,
                topic=event.topic,
                title=event.title,
                url=event.url,
                reply_to=event.reply_to,
                status="prep",
            )
        store.enqueue(
            event.topic, render.message(event, summary, drafting), url=event.url, done_job=job.id
        )
        _LOGGER.info("reported %s%s", event.key, " (drafting)" if drafting else "")

    # -- drafts -----------------------------------------------------------------

    def draft(self, draft: Draft) -> None:
        store = self.store
        label = f"{draft.repo}#{draft.number}"
        self.busy = label
        self.between("starting")
        why = ""
        try:
            result = drafts.generate(
                self.cfg, draft, self.history, self.root, self.files, beat=self.between
            )
        except drafts.Unfit as err:
            result, why = None, f": {err}"
        finally:
            self.busy = ""
        if result is None:
            store.fail_draft(draft.id)
            store.enqueue(
                draft.topic,
                render.system(f"The draft reply to {label} failed{why}."),
                url=draft.url,
            )
            return
        version = store.finish_draft(
            draft.id, result.reply, result.note, result.attachments, result.verdict.to_json()
        )
        ready = store.draft(draft.id)
        # The assessment first, silently: the draft right after it is what needs you.
        store.enqueue(
            draft.topic, render.verdict(ready, result.verdict), url=draft.url, silent=True
        )
        drafts.offer(store, ready, version)
        _LOGGER.info("draft %d ready for %s", draft.id, label)

    # -- briefs -----------------------------------------------------------------

    def brief(self, job: Job) -> None:
        """Write a brief from the code the watcher fetched (the release's tag, else the
        default branch) and put it up for approval."""

        repo, ref, label, tag = (job.payload[k] for k in ("repo", "ref", "label", "tag"))
        self.between("writing the brief")
        releases = self.history.releases(repo)
        release = next((r for r in releases if r.tag == tag), None) if tag else None
        if tag:
            target = snapshot.version_path(self.root, repo, tag)
        else:
            target = snapshot.path_for(self.root, repo)
        text = None
        if target is None or not target.is_dir():
            _LOGGER.warning("brief %s %s: no code copy at %s", repo, label, target)
        elif tag is None:
            text = brief.generate(self.cfg, repo, target)
        elif release is not None:
            text = brief.generate(self.cfg, repo, target, release, releases)
        brief_id = self.history.add_brief(repo, ref, label, text)
        if text is None:
            message = f"Brief for {repo} {label} failed; retrying in a day."
            self.store.enqueue("system", render.system(message), done_job=job.id)
            return
        self.store.enqueue(
            "system",
            render.brief(repo, label, text),
            buttons=[
                ("✅ Use it", f"brief:approve:{brief_id}"),
                ("🗑 Discard", f"brief:reject:{brief_id}"),
            ],
            done_job=job.id,
        )

    # -- replays ----------------------------------------------------------------

    def replay(self, job: Job) -> None:
        """Replay closed issues (``evaluate``); the watcher fetched their files and code."""

        p = job.payload
        numbers = [tuple(n) if isinstance(n, list) else n for n in p["numbers"]]
        out = Path(p["out"])

        def say(text: str) -> None:
            _LOGGER.info("replay %s: %s", p["repo"], text)
            self.between(text)

        outcomes = evaluate.run(
            self.cfg, self.history, p["repo"], self.root, self.files, out, numbers, p["limit"], say
        )
        text = f"Replay of {p['repo']} done: {len(outcomes)} issue(s), report in {out}."
        self.store.enqueue("system", render.system(text), done_job=job.id)


def wants_draft(event: Event, summary: llm.Summary | None, cfg: Config) -> bool:
    """A stranger's new issue or discussion, or a stranger's comment on one that the
    summary says needs a reply. A new thread doesn't depend on the small model's
    one-line judgement (a report answering an earlier question reads as needing no
    reply): the assessment judges it, and the user can reject the draft."""

    opening = event.kind in OPENING_KINDS
    return (
        cfg.drafts
        and (opening or (summary is not None and summary.needs_reply))
        and event.thread_kind is not None
        and not event.is_bot
        and event.association not in MAINTAINERS
    )


def _check_model(cfg: Config) -> str:
    if not cfg.summary_model:
        return "summaries off (no llm.summary_model)"
    try:
        models = llm.available_models(cfg)
    except Exception as err:  # noqa: BLE001
        return (
            f"Ollama unreachable at {cfg.llm_url} ({type(err).__name__});"
            " reporting without summaries"
        )
    if not llm.has_model(models, cfg.summary_model):
        return f"model {cfg.summary_model} not in `ollama list`; reporting without summaries"
    return f"summaries by {cfg.summary_model}"


def run(cfg: Config) -> None:
    store = Store(DATA_DIR / "watchtower.db")
    history = History(DATA_DIR / "history.db")
    worker = Worker(cfg, store, history, DATA_DIR / "repos", DATA_DIR / "attachments")
    store.forget_beat("drafter")  # this process's name before it did all model work
    store.requeue_jobs()
    store.requeue_drafting()
    drafting = f"drafts by {cfg.agent_model}" if cfg.drafts else "drafts off"
    status = f"{_check_model(cfg)}; {drafting}"
    _LOGGER.info("worker started: %s", status)
    store.enqueue("system", render.system(f"Worker online: {status}."))
    while True:
        store.beat("worker", "idle")
        if not worker.step():
            time.sleep(IDLE_SECONDS)
