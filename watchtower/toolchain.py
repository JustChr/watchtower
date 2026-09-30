"""The toolchain process: installs a repo's dependencies for the runner to use.

The runner has no network, so somebody with a network has to install what a repo's
checks need. That is this service -- and it runs only what the repo's **default branch**
says (the ``setup`` steps of ``gates``: ``pip install -r ...``, ``npm ci``), never a PR's
code. It has a network and nothing else: no secret, no ``/data``, gVisor like the runner.

Its world is ``/toolchain`` (jobs from the watcher, results back: the same protocol as the
runner's, ``sandbox``) and ``/tools`` (the installed environments, which it writes and the
runner only reads). A job is a copy of the default branch and the steps: the first makes a
virtualenv at ``$VIRTUAL_ENV``, the others install into it. ``node_modules`` (an ``npm ci``
lands in the tree) is moved beside it. The environment is swapped in only when every step
passed, marked with the job's key -- a hash of the dependency files -- so the watcher knows
whether it is current. Like the runner it takes one job and exits: compose restarts it.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from . import gates, runner, sandbox

_LOGGER = logging.getLogger(__name__)

TOOLCHAIN_DIR = Path(os.environ.get("WATCHTOWER_TOOLCHAIN", "/toolchain"))
KEY_FILE = ".key"
MODULES = "node_modules"
CREATE = gates.Step("create the environment", 'python3 -m venv "$VIRTUAL_ENV"')
# The files that decide what gets installed: a change to one makes the environment stale.
_DEPENDENCY_FILES = frozenset(
    {
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "pyproject.toml",
        "poetry.lock",
        "uv.lock",
        "pipfile",
        "pipfile.lock",
        "setup.py",
        "setup.cfg",
        "tox.ini",
    }
)
_SKIP = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv"})
MAX_KEYED = 200  # dependency files hashed


def _is_dependency_file(name: str) -> bool:
    lowered = name.lower()
    return lowered in _DEPENDENCY_FILES or (
        lowered.startswith("requirements") and lowered.endswith((".txt", ".in"))
    )


def tools_key(tree: Path, setup: Sequence[gates.Step]) -> str:
    """A hash of what the install depends on: the setup commands and the dependency files
    of ``tree`` (the default branch). The same key, the same environment."""

    digest = hashlib.sha256()
    for step in setup:
        digest.update(f"{step.workdir}\0{step.script}\0".encode())
    found = 0
    for folder, dirs, files in os.walk(tree):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP)
        for name in sorted(files):
            path = Path(folder) / name
            if _is_dependency_file(name) and not path.is_symlink() and found < MAX_KEYED:
                found += 1
                digest.update(path.relative_to(tree).as_posix().encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()[:32]


def env_path(tools_root: Path, repo: str) -> Path | None:
    """Where ``repo``'s environment lives; ``None`` for a name that isn't ``owner/name``."""

    if not runner._REPO.fullmatch(repo) or ".." in repo.split("/"):
        return None
    return tools_root.joinpath(*repo.split("/"))


def current_key(tools_root: Path, repo: str) -> str | None:
    """The key the installed environment was made with, if there is one."""

    path = env_path(tools_root, repo)
    marker = path / KEY_FILE if path else None
    try:
        return marker.read_text(encoding="utf-8").strip() if marker else None
    except OSError:
        return None


def setup_job(job_id: str, repo: str, key: str, setup: Sequence[gates.Step]) -> sandbox.Job:
    """The job that builds an environment: create it, then the repo's install steps. The
    job's ``sha`` is the key. Installs get the time the checks get."""

    return sandbox.Job(job_id, repo, 0, key, (CREATE, *setup))


def install(
    where: Path,
    tools_root: Path,
    *,
    shell: Sequence[str] = sandbox.SHELL,
    sweep: Callable[[], None] | None = sandbox.sweep,
) -> sandbox.Result:
    """Run the setup job in ``where``; when every step passed, swap the new environment in
    and publish the result. Publishing comes last: the watcher moves on once it sees it."""

    try:
        job = sandbox.Job.from_json((where / sandbox.JOB_FILE).read_text(encoding="utf-8"))
    except OSError, ValueError, TypeError, KeyError:
        return sandbox.execute(where, None, shell=shell, finish=sweep)  # says it's unreadable
    final = env_path(tools_root, job.repo)
    if final is None:
        result = sandbox.Result(job.id, (), error="not a repository name")
        sandbox.publish(where, result)
        return result
    new = final.with_name(final.name + ".new")
    shutil.rmtree(new, ignore_errors=True)
    new.parent.mkdir(parents=True, exist_ok=True)
    result = sandbox.execute(where, new, shell=shell, finish=sweep, publish=False)
    if result.green:
        try:
            _swap(where / sandbox.TREE, new, final, job.sha)
        except OSError as err:
            result = sandbox.Result(
                job.id, result.steps, result.not_run, result.seconds, f"install failed: {err}"[:300]
            )
    if not result.green:
        shutil.rmtree(new, ignore_errors=True)
    sandbox.publish(where, result)
    return result


def _swap(tree: Path, new: Path, final: Path, key: str) -> None:
    modules = tree / MODULES
    if modules.is_dir() and not modules.is_symlink():
        shutil.move(str(modules), new / MODULES)
    (new / KEY_FILE).write_text(key, encoding="utf-8")
    old = final.with_name(final.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    new.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def run(
    root: Path = TOOLCHAIN_DIR,
    tools_root: Path = runner.TOOLS_DIR,
    *,
    idle: float = runner.IDLE_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    shell: Sequence[str] = sandbox.SHELL,
    sweep: Callable[[], None] | None = sandbox.sweep,
    waits: int | None = None,
) -> str | None:
    """Wait for a setup job, run it and return its id (see ``runner.run``)."""

    idled = 0
    while True:
        where = sandbox.next_job(root)
        if where is not None:
            break
        if waits is not None and idled >= waits:
            return None
        idled += 1
        sleep(idle)
    _LOGGER.info("installing for job %s", where.name)
    result = install(where, tools_root, shell=shell, sweep=sweep)
    _LOGGER.info("job %s done: %s", where.name, "green" if result.green else "failed")
    return where.name
