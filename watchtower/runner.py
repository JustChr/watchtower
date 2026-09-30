"""The runner process: runs a PR's checks, one job per container.

It holds nothing: no config, no secret, no network, no ``/data``. Its whole world is
``/sandbox`` (the watcher puts a job there and reads its result) and ``/tools`` (the
repos' installed environments, read-only). It takes one job, runs it (``sandbox``),
and exits: compose restarts it (``restart: always``) with a fresh filesystem, so
nothing a stranger's code leaves behind reaches the next job.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from . import sandbox

_LOGGER = logging.getLogger(__name__)

SANDBOX_DIR = Path(os.environ.get("WATCHTOWER_SANDBOX", "/sandbox"))
TOOLS_DIR = Path(os.environ.get("WATCHTOWER_TOOLS", "/tools"))
IDLE_SECONDS = 2.0
_REPO = re.compile(r"[\w.-]+/[\w.-]+")


def tools_for(tools_root: Path, repo: str) -> Path | None:
    """The installed environment of ``repo`` (``<owner>/<name>`` under ``tools_root``), if
    the toolchain service made one."""

    if not _REPO.fullmatch(repo) or ".." in repo.split("/"):
        return None
    path = tools_root.joinpath(*repo.split("/"))
    return path if path.is_dir() and path.resolve().is_relative_to(tools_root.resolve()) else None


def run(
    root: Path = SANDBOX_DIR,
    tools_root: Path = TOOLS_DIR,
    *,
    idle: float = IDLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    finish: Callable[[], None] | None = sandbox.sweep,
    shell: Sequence[str] = sandbox.SHELL,
    waits: int | None = None,
) -> str | None:
    """Wait for a job, run it and return its id. ``waits``: give up (``None``) after that
    many idle waits; the default is to wait for ever."""

    idled = 0
    while True:
        where = sandbox.next_job(root)
        if where is not None:
            break
        if waits is not None and idled >= waits:
            return None
        idled += 1
        sleep(idle)
    try:
        repo = sandbox.Job.from_json((where / sandbox.JOB_FILE).read_text(encoding="utf-8")).repo
    except OSError, ValueError, TypeError, KeyError:
        repo = ""  # ``execute`` reports the unreadable job
    _LOGGER.info("running job %s", where.name)
    result = sandbox.execute(where, tools_for(tools_root, repo), shell=shell, finish=finish)
    _LOGGER.info(
        "job %s done: %s",
        where.name,
        "green" if result.green else result.error or "not green",
    )
    return where.name
