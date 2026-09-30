"""Running a PR's checks in the sandbox: the job going in, the result coming out.

Two sides share this module and a volume (``/sandbox``), and nothing else:

- the **watcher** (trusted, has the read token) writes a job -- the merged tree and
  the steps to run, from ``gates`` -- and reads the result back;
- the **runner** (no network, no secrets, gVisor, one job per container) executes the
  job and writes the result.

The tree and every command's output are strangers' text and the run is their code,
so the result is untrusted on the way back: ``parse_result`` accepts one fixed shape,
with every size capped, and anything else is no result. The runner never sees ``/data``.

A run can't forge its own result: the supervisor keeps the outcome in memory and writes
``result.json`` only after killing everything the steps started (``sweep``), replacing
the file atomically. And a result is only ever an input to a review that you approve.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from . import gates

JOBS = "jobs"
JOB_FILE = "job.json"
RESULT_FILE = "result.json"
TREE = "tree"
ENV_DIR = "env"  # the toolchain service's installs, per repo: mounted read-only

STEP_SECONDS = 900
TOTAL_SECONDS = 2400
MAX_OUTPUT = 4000  # characters of a step's output kept: its end
MAX_FILE = 50 * 1024 * 1024  # bytes one step may write to a file
MAX_RESULT = 100_000  # bytes of a result file read
MAX_STEPS = gates.MAX_STEPS
MAX_NAME = 120
MAX_SCRIPT = 2000
PASSED, FAILED, TIMEOUT, SKIPPED = "passed", "failed", "timeout", "skipped"
STATUSES = frozenset({PASSED, FAILED, TIMEOUT, SKIPPED})
SHELL = ("bash", "-e", "-o", "pipefail", "-c")


@dataclass(frozen=True)
class Job:
    id: str
    repo: str
    number: int
    sha: str  # the merge commit the tree is
    steps: tuple[gates.Step, ...]
    not_run: tuple[tuple[str, str], ...] = ()  # what the plan says can't run: (name, reason)
    python: str = ""
    node: str = ""
    step_seconds: int = STEP_SECONDS
    total_seconds: int = TOTAL_SECONDS

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Job:
        data = json.loads(text)
        data["steps"] = tuple(gates.Step(**s) for s in data["steps"])
        data["not_run"] = tuple(tuple(n) for n in data.get("not_run", ()))
        return cls(**data)


@dataclass(frozen=True)
class StepResult:
    name: str
    script: str
    status: str
    code: int | None = None  # the exit code, when it ran to the end
    seconds: float = 0.0
    output: str = ""  # the end of its output, capped


@dataclass(frozen=True)
class Result:
    job: str
    steps: tuple[StepResult, ...]
    not_run: tuple[tuple[str, str], ...] = ()
    seconds: float = 0.0
    error: str = ""  # the job couldn't run at all

    @property
    def green(self) -> bool:
        return not self.error and bool(self.steps) and all(s.status == PASSED for s in self.steps)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# -- the watcher's side ----------------------------------------------------------------------


def job_dir(root: Path, job_id: str) -> Path:
    if not job_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("a job id is letters, digits, - and _")
    return root / JOBS / job_id


def write_job(root: Path, job: Job, tree: Path) -> Path:
    """Put a job where the runner takes it from: the tree first, ``job.json`` last (the
    runner starts only when it exists). Returns the job's folder."""

    where = job_dir(root, job.id)
    shutil.rmtree(where, ignore_errors=True)
    where.mkdir(parents=True)
    shutil.copytree(tree, where / TREE, symlinks=True)
    tmp = where / (JOB_FILE + ".new")
    tmp.write_text(job.to_json(), encoding="utf-8")
    tmp.replace(where / JOB_FILE)
    return where


def _text(value: object, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def parse_result(raw: bytes | str, job_id: str) -> Result | None:
    """The runner's result, or ``None`` if it isn't exactly the agreed shape. It came
    out of a sandbox that ran a stranger's code: nothing in it is believed beyond that."""

    if len(raw) > MAX_RESULT:
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("job") != job_id:
        return None
    listed = data.get("steps")
    if not isinstance(listed, list) or len(listed) > MAX_STEPS:
        return None
    steps = []
    for item in listed:
        if not isinstance(item, dict):
            return None
        name, script = _text(item.get("name"), MAX_NAME), _text(item.get("script"), MAX_SCRIPT)
        output = _text(item.get("output"), MAX_OUTPUT)
        code, took = item.get("code"), item.get("seconds")
        if (
            name is None
            or script is None
            or output is None
            or item.get("status") not in STATUSES
            or not (code is None or (isinstance(code, int) and not isinstance(code, bool)))
            or not isinstance(took, int | float)
            or isinstance(took, bool)
        ):
            return None
        steps.append(StepResult(name, script, item["status"], code, float(took), output))
    not_run = []
    for item in data.get("not_run") if isinstance(data.get("not_run"), list) else []:
        if isinstance(item, list) and len(item) == 2 and all(isinstance(x, str) for x in item):
            not_run.append((item[0][:MAX_NAME], item[1][: MAX_NAME * 2]))
    seconds = data.get("seconds")
    return Result(
        job_id,
        tuple(steps),
        tuple(not_run[:MAX_STEPS]),
        float(seconds)
        if isinstance(seconds, int | float) and not isinstance(seconds, bool)
        else 0.0,
        _text(data.get("error"), 300) or "",
    )


def read_result(root: Path, job_id: str) -> Result | None:
    path = job_dir(root, job_id) / RESULT_FILE
    if path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_RESULT + 1)
    except OSError:
        return None
    return parse_result(raw, job_id)


def finished(root: Path, job_id: str) -> bool:
    return (job_dir(root, job_id) / RESULT_FILE).exists()


# -- the runner's side -----------------------------------------------------------------------


def next_job(root: Path) -> Path | None:
    """The oldest job that has its ``job.json`` and no result yet."""

    folder = root / JOBS
    if not folder.is_dir():
        return None
    waiting = [
        p
        for p in folder.iterdir()
        if (p / JOB_FILE).is_file() and not (p / RESULT_FILE).exists() and not p.is_symlink()
    ]
    return min(waiting, key=lambda p: (p / JOB_FILE).stat().st_mtime, default=None)


def clean_env(where: Path, tools: Path | None, home: Path) -> dict[str, str]:
    """What a step's environment is: built from nothing, so no secret, token or host
    variable of the runner reaches a stranger's code."""

    path = os.defpath if os.name == "nt" else "/usr/local/bin:/usr/bin:/bin"
    if tools is not None:
        path = os.pathsep.join([str(tools / "bin"), str(tools / "node_modules" / ".bin"), path])
    env = {"PATH": path, "HOME": str(home), "CI": "true", "LANG": "C.UTF-8", "TMPDIR": str(home)}
    if tools is not None:
        env["VIRTUAL_ENV"] = str(tools)
        env["NODE_PATH"] = str(tools / "node_modules")
    return env


def _limit_files() -> None:  # pragma: no cover -- runs in the child, POSIX only
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))


def _kill(proc: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError, PermissionError:
        pass


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - MAX_OUTPUT * 4))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")[-MAX_OUTPUT:]


def run_step(
    step: gates.Step,
    tree: Path,
    env: dict[str, str],
    seconds: float,
    shell: Sequence[str] = SHELL,
) -> StepResult:
    """One command, in its folder of the tree, with a clock: killed with everything it
    started when the time is up."""

    base = tree.resolve()
    workdir = (base / gates.safe_workdir(step.workdir)).resolve()
    if not workdir.is_relative_to(base) or not workdir.is_dir():
        return StepResult(step.name, step.script[:MAX_SCRIPT], SKIPPED, output="no such folder")
    started = time.monotonic()
    with tempfile.NamedTemporaryFile("w+b", prefix="out-", delete=False) as out:
        log = Path(out.name)
        try:
            proc = subprocess.Popen(
                [*shell, step.script],
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=hasattr(os, "killpg"),
                preexec_fn=_limit_files if os.name == "posix" else None,
            )
        except OSError as err:
            log.unlink(missing_ok=True)
            return StepResult(step.name, step.script[:MAX_SCRIPT], SKIPPED, output=str(err)[:200])
        try:
            code = proc.wait(timeout=seconds)
            status = PASSED if code == 0 else FAILED
        except subprocess.TimeoutExpired:
            code, status = None, TIMEOUT
        finally:
            _kill(proc)  # whatever it left running
            proc.wait()
    output = _tail(log)
    log.unlink(missing_ok=True)
    return StepResult(
        step.name[:MAX_NAME],
        step.script[:MAX_SCRIPT],
        status,
        code,
        round(time.monotonic() - started, 1),
        output,
    )


def sweep() -> None:  # pragma: no cover -- kills the runner's whole container
    """Kill every other process this user has: nothing a step started may outlive it
    to touch the result. Only for the runner, where the container is one job."""

    try:
        os.kill(-1, signal.SIGKILL)
    except ProcessLookupError, PermissionError, OSError:
        pass


def execute(
    where: Path,
    tools: Path | None = None,
    *,
    shell: Sequence[str] = SHELL,
    finish: Callable[[], None] | None = None,
) -> Result:
    """Run the job in ``where`` and write its result. ``tools``: the repo's installed
    environment (read-only). ``finish`` runs after the last step, before the result is
    written -- the runner passes ``sweep``."""

    try:
        job = Job.from_json((where / JOB_FILE).read_text(encoding="utf-8"))
        if job.id != where.name:
            raise ValueError("the job isn't in its own folder")
        job_dir(where.parent.parent, job.id)  # a sane id
    except OSError, ValueError, TypeError, KeyError:
        result = Result(where.name, (), error="the job file is unreadable")
        _write(where, result, finish)
        return result
    tree = where / TREE
    started = time.monotonic()
    steps: list[StepResult] = []
    with tempfile.TemporaryDirectory(prefix="home-") as home:
        env = clean_env(tree, tools, Path(home))
        for step in job.steps[:MAX_STEPS]:
            left = job.total_seconds - (time.monotonic() - started)
            if left <= 0:
                steps.append(
                    StepResult(step.name, step.script[:MAX_SCRIPT], SKIPPED, output="out of time")
                )
                continue
            steps.append(run_step(step, tree, env, min(job.step_seconds, left), shell))
    result = Result(job.id, tuple(steps), job.not_run, round(time.monotonic() - started, 1))
    _write(where, result, finish)
    return result


def _write(where: Path, result: Result, finish: Callable[[], None] | None) -> None:
    if finish is not None:
        finish()
    tmp = where / (RESULT_FILE + ".new")
    tmp.write_text(result.to_json(), encoding="utf-8")
    tmp.replace(where / RESULT_FILE)


def busy(root: Path) -> bool:
    """Whether a job is in the sandbox already. The runner shares its volume with the code
    it runs, so a second job's tree could be tampered with by the first: one at a time."""

    folder = root / JOBS
    return folder.is_dir() and any(folder.iterdir())


def remove_job(root: Path, job_id: str) -> None:
    shutil.rmtree(job_dir(root, job_id), ignore_errors=True)


# -- proving the isolation on the box --------------------------------------------------------

_NO_NETWORK = (
    'python3 -c "import socket, sys\n'
    "try:\n    socket.create_connection(('1.1.1.1', 53), 3)\n"
    'except OSError:\n    sys.exit(0)\nsys.exit(1)"'
)
# Each probe passes when the isolation holds. The last one only shows what it sees.
SELFTEST_STEPS = (
    gates.Step("runs a command", "echo hello"),
    gates.Step("a failing command is reported as failed", "exit 3"),
    gates.Step("no network", _NO_NETWORK),
    gates.Step("no /data", "test ! -e /data"),
    gates.Step("no secrets", 'test ! -e /run/secrets/github_read && test -z "$GITHUB_TOKEN"'),
    gates.Step("read-only root", "if touch /usr/probe 2>/dev/null; then exit 1; fi"),
    gates.Step("cannot see the config", "test ! -e /config"),
    gates.Step("kernel (should say gVisor)", "dmesg 2>&1 | head -1 || true"),
)


def selftest_job() -> Job:
    return Job("selftest", "owner/repo", 0, "0" * 40, SELFTEST_STEPS, step_seconds=30)
