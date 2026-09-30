"""The watcher's half of running a PR's checks (``sandbox`` has the protocol).

For a review draft waiting in ``prep``, ``settle`` moves its ``gates`` stage along:

- the plan (``gates.plan`` of the default branch) has nothing to run -> ``none``;
- the PR doesn't merge into its base -> ``conflict`` (the review asks for a rebase);
- GitHub won't say, or the merged tree can't be fetched -> ``unknown``;
- another job is in the sandbox -> ``waiting`` (one at a time);
- the repo's dependencies aren't installed (or are stale) -> a setup job for the toolchain
  service, ``setting-up``, then ``setup-failed`` if they can't be installed;
- else the merged tree and the steps become a job -> ``running``, then ``done`` with the
  runner's result, or ``timeout`` if it never comes.

Every state but ``waiting`` and ``running`` is final: the draft then goes to the worker,
whose review says what ran and what didn't. Nothing here runs a command.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from pathlib import Path

from . import gates, pr, sandbox, snapshot, toolchain
from .store import Draft, Store

_LOGGER = logging.getLogger(__name__)

STAGE = "gates"
WAIT_SECONDS = sandbox.TOTAL_SECONDS + 900  # a result later than this isn't coming
OPEN = frozenset({"waiting", "setting-up", "running"})


def state(store: Store, draft: Draft) -> dict | None:
    return store.stages(draft.id).get(STAGE)


def _put(store: Store, draft: Draft, **data: object) -> None:
    store.put_stage(draft.id, STAGE, data)


def settle(
    draft: Draft,
    source: pr.Source,
    store: Store,
    root: Path,
    box: Path,
    *,
    toolbox: Path | None = None,
    tools: Path | None = None,
    now: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Advance the draft's checks; ``True`` once they have settled. ``root``: the code
    snapshots (``/data/repos``), ``box``: the sandbox volume; ``toolbox`` and ``tools``: the
    toolchain service's jobs and the environments it installs (without them, nothing is
    installed here and the runner uses what is there)."""

    now = time.time() if now is None else now
    current = state(store, draft)
    if current and current["state"] not in OPEN:
        return True
    if current and current["state"] == "running":
        job = current["job"]
        result = sandbox.read_result(box, job)
        if result is not None:
            _put(store, draft, **{**current, "state": "done", "result": dataclasses.asdict(result)})
            sandbox.remove_job(box, job)
            return True
        if now - current["started"] > WAIT_SECONDS:
            _put(store, draft, **{**current, "state": "timeout"})
            sandbox.remove_job(box, job)
            return True
        return False
    installed = False
    if current and current["state"] == "setting-up" and toolbox is not None:
        job = current["setup_job"]
        result = sandbox.read_result(toolbox, job)
        if result is None:
            if now - current["started"] > WAIT_SECONDS:
                _put(store, draft, **{**current, "state": "setup-failed", "result": None})
                sandbox.remove_job(toolbox, job)
                return True
            return False
        sandbox.remove_job(toolbox, job)
        if not result.green:
            _put(
                store,
                draft,
                **{
                    **current,
                    "state": "setup-failed",
                    "job": job,
                    "result": dataclasses.asdict(result),
                },
            )
            return True
        installed = True  # the dependencies are in: on to the checks

    main = snapshot.path_for(root, draft.repo)
    plan = gates.plan(main) if main.is_dir() else gates.Plan()
    if not plan:
        _put(store, draft, state="none", reason="the repo's CI and conventions name no checks")
        return True
    info = {
        "source": plan.source,
        "not_run": [list(n) for n in plan.not_run],
        "setup": [dataclasses.asdict(s) for s in plan.setup],
    }
    try:
        merged = pr.fetch_merged(source, draft.repo, draft.number, root, sleep=sleep)
    except Exception as err:  # noqa: BLE001 -- the review says the checks didn't run
        _LOGGER.warning("%s#%d: merged tree: %s", draft.repo, draft.number, err)
        _put(store, draft, state="unknown", reason="the merged tree couldn't be fetched", **info)
        return True
    if merged is None:
        facts = pr.load(root, draft.repo, draft.number) or {}
        if facts.get("mergeable") is False:
            _put(store, draft, state="conflict", reason="it doesn't merge into its base", **info)
        else:
            _put(store, draft, state="unknown", reason="GitHub couldn't say if it merges", **info)
        return True

    if plan.setup and toolbox is not None and tools is not None and not installed:
        key = toolchain.tools_key(main, plan.setup)
        if toolchain.current_key(tools, draft.repo) != key:
            if sandbox.busy(toolbox):
                if not current:
                    _put(store, draft, state="waiting", **info)
                return False
            setup = toolchain.setup_job(f"t{draft.id}", draft.repo, key, plan.setup)
            sandbox.write_job(toolbox, setup, main)
            _put(store, draft, state="setting-up", setup_job=setup.id, started=now, **info)
            return False
    if sandbox.busy(box):
        if not current:
            _put(store, draft, state="waiting", **info)
        return False
    job = sandbox.Job(
        id=f"d{draft.id}",
        repo=draft.repo,
        number=draft.number,
        sha=(merged.parent / "merged.sha").read_text(encoding="utf-8"),
        steps=plan.gates,
        not_run=plan.not_run,
        python=plan.python,
        node=plan.node,
    )
    sandbox.write_job(box, job, merged)
    _put(store, draft, state="running", job=job.id, started=now, **info)
    return False
