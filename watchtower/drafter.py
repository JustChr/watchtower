"""The drafter process: writes reply drafts with the big local model.

It is the one process that feeds strangers' text to the agent model, so it holds
no secret and has no internet: its only network is ``watchtower-llm`` to Ollama.
It reads the history, the code snapshot and the attachments the watcher
downloaded, and writes drafts to the store; the gateway shows them, the poster
posts them after approval.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from . import drafts, render
from .config import DATA_DIR, Config
from .history import History
from .store import Draft, Store

_LOGGER = logging.getLogger(__name__)

IDLE_SECONDS = 5


def work(
    draft: Draft,
    cfg: Config,
    store: Store,
    history: History,
    root: Path,
    folder: Path | None = None,
) -> None:
    label = f"{draft.repo}#{draft.number}"

    def beat(step: str) -> None:
        # Between passes: a draft takes several model calls, each up to agent_timeout.
        store.beat("drafter", f"{label}: {step}")

    beat("starting")
    why = ""
    try:
        result = drafts.generate(cfg, draft, history, root, folder, beat=beat)
    except drafts.Unfit as err:
        result, why = None, f": {err}"
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
    store.enqueue(draft.topic, render.verdict(ready, result.verdict), url=draft.url, silent=True)
    drafts.offer(store, ready, version)
    _LOGGER.info("draft %d ready for %s#%d", draft.id, draft.repo, draft.number)


def run(cfg: Config) -> None:
    store = Store(DATA_DIR / "watchtower.db")
    history = History(DATA_DIR / "history.db")
    root = DATA_DIR / "repos"
    store.requeue_drafting()
    status = f"drafts by {cfg.agent_model}" if cfg.drafts else "drafts off"
    _LOGGER.info("drafter started: %s", status)
    store.enqueue("system", render.system(f"Drafter online: {status}."))
    while True:
        store.beat("drafter", "idle")
        draft = store.claim_draft() if cfg.drafts else None
        if draft is None:
            time.sleep(IDLE_SECONDS)
            continue
        try:
            work(draft, cfg, store, history, root, DATA_DIR / "attachments")
        except Exception:  # noqa: BLE001 -- one bad draft mustn't stop the drafter
            _LOGGER.exception("draft %d: unexpected error", draft.id)
            store.fail_draft(draft.id)
