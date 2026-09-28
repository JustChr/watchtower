"""The drafter process: writes reply drafts with the big local model.

It is the one process that feeds strangers' text to the agent model, so it holds
no secret and has no internet: its only network is ``watchtower-llm`` to Ollama.
It reads the history and the code snapshot, and writes drafts to the store; the
gateway shows them, the poster posts them after approval.
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


def work(draft: Draft, cfg: Config, store: Store, history: History, root: Path) -> None:
    store.beat("drafter", f"drafting {draft.repo}#{draft.number}")  # can take minutes
    result = drafts.generate(cfg, draft, history, root)
    if result is None:
        store.fail_draft(draft.id)
        store.enqueue(
            draft.topic,
            render.system(f"The draft reply to {draft.repo}#{draft.number} failed."),
            url=draft.url,
        )
        return
    text, note = result
    version = store.finish_draft(draft.id, text, note)
    drafts.offer(store, store.draft(draft.id), version)
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
            work(draft, cfg, store, history, root)
        except Exception:  # noqa: BLE001 -- one bad draft mustn't stop the drafter
            _LOGGER.exception("draft %d: unexpected error", draft.id)
            store.fail_draft(draft.id)
