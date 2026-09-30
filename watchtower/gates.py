"""What "the checks" are for a repo, read from the repo itself.

A repo's own CI says what has to pass before a change is merged, so Watchtower reads
that instead of being told per repo: the ``run:`` steps of the workflows in
``.github/workflows/`` on the **default branch** (the maintainers' text, not a PR's)
that run for pull requests. Code does the reading, not the model.

- an install step (``pip install``, ``npm ci``, ...) is **setup**;
- any other ``run:`` step is a **gate**;
- a step this box can't or shouldn't run -- a third-party action (hassfest, HACS), a
  secret, a deploy, the network, an expression we can't fill in -- is **not run**,
  with the reason, so a review never implies it passed.

A repo without such a workflow falls back to its conventions (``pyproject.toml``,
``package.json`` scripts, a ``Makefile``). ``.watchtower/gates.toml`` on the default
branch overrides both, for a repo whose CI can't be mapped.

This only plans: nothing here runs a command.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml

WORKFLOWS = ".github/workflows"
OVERRIDE = ".watchtower/gates.toml"
MAX_WORKFLOW = 200_000  # bytes of one workflow file read
MAX_STEPS = 60

# Install steps: run where there is a network (the toolchain service), from main.
_SETUP = re.compile(
    r"^\s*(?:python\d?(?:\.\d+)?\s+-m\s+)?"
    r"(?:pip3?\s+install|npm\s+(?:ci|install|i)\b|yarn\s+install|pnpm\s+install|"
    r"poetry\s+install|uv\s+(?:sync|pip\s+install)|pipenv\s+install|bundle\s+install|"
    r"go\s+mod\s+download|cargo\s+fetch)",
    re.IGNORECASE,
)
# Things a check has no business doing in a sandbox without a network or credentials.
_OUTSIDE = re.compile(
    r"\b(?:git\s+push|gh\s+\w+|docker|curl|wget|twine|npm\s+publish|aws|gcloud|az\s+\w+|"
    r"sudo|apt(?:-get)?\s+install|brew)\b",
    re.IGNORECASE,
)
_EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)
_SECRET = re.compile(r"\bsecrets\.|\bGITHUB_TOKEN\b|\bgithub\.token\b", re.IGNORECASE)
_MATRIX = re.compile(r"^\s*matrix\.([\w-]+)\s*$")
_TOOLING = {"actions/checkout", "actions/cache", "actions/upload-artifact"}


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    workdir: str = "."
    job: str = ""


@dataclass(frozen=True)
class Plan:
    """What to install, what to run, and what can't be run here."""

    setup: tuple[Step, ...] = ()
    gates: tuple[Step, ...] = ()
    not_run: tuple[tuple[str, str], ...] = ()  # (name, reason)
    python: str = ""  # the version the workflow asks for, if it says
    node: str = ""
    source: str = ""  # where the plan came from: workflows, conventions, override

    def __bool__(self) -> bool:
        return bool(self.gates)


def _runs_for_prs(triggers: object) -> bool:
    """Whether a workflow's ``on:`` includes ``pull_request`` (``pull_request_target``
    runs with secrets and the base's rights: never)."""

    if isinstance(triggers, str):
        return triggers == "pull_request"
    if isinstance(triggers, list):
        return "pull_request" in triggers
    return isinstance(triggers, dict) and "pull_request" in triggers


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _matrix_first(job: dict) -> dict[str, str]:
    """The first value of each simple matrix axis: the one run standing for the rest."""

    matrix = (job.get("strategy") or {}).get("matrix")
    first: dict[str, str] = {}
    if isinstance(matrix, dict):
        for key, values in matrix.items():
            if isinstance(values, list) and values and isinstance(values[0], str | int | float):
                first[str(key)] = str(values[0])
    return first


def _fill(text: str, matrix: dict[str, str]) -> str | None:
    """``text`` with ``${{ matrix.x }}`` filled in; ``None`` if another expression is left."""

    def one(found: re.Match) -> str:
        axis = _MATRIX.match(found[1])
        return matrix.get(axis[1], found[0]) if axis else found[0]

    filled = _EXPRESSION.sub(one, text)
    return None if "${{" in filled else filled


def from_workflow(text: str, name: str = "") -> Plan | None:
    """The plan of one workflow file, or ``None`` if it isn't for pull requests (or isn't
    a workflow at all)."""

    try:
        doc = yaml.safe_load(text[:MAX_WORKFLOW])
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None
    triggers = doc.get("on", doc.get(True))  # YAML 1.1 reads a bare ``on`` as True
    jobs = doc.get("jobs")
    if not _runs_for_prs(triggers) or not isinstance(jobs, dict):
        return None
    setup: list[Step] = []
    gates: list[Step] = []
    not_run: list[tuple[str, str]] = []
    python = node = ""
    for job_id, job in jobs.items():
        if not isinstance(job, dict) or not isinstance(job.get("steps"), list):
            continue
        matrix = _matrix_first(job)
        if job.get("environment"):
            not_run.append((f"job {job_id}", "it deploys to an environment"))
            continue
        for number, step in enumerate(job["steps"][:MAX_STEPS], 1):
            if not isinstance(step, dict):
                continue
            label = _text(step.get("name")) or f"{job_id} step {number}"
            uses, run = _text(step.get("uses")), _text(step.get("run"))
            options = step.get("with") if isinstance(step.get("with"), dict) else {}
            if uses:
                action = uses.split("@")[0]
                if action == "actions/setup-python":
                    python = python or str(options.get("python-version", ""))
                elif action == "actions/setup-node":
                    node = node or str(options.get("node-version", ""))
                elif action not in _TOOLING and not action.startswith("actions/setup-"):
                    not_run.append((label, f"it uses the action {action}"))
                continue
            if not run.strip():
                continue
            if _SECRET.search(run) or _SECRET.search(str(step.get("env", ""))):
                not_run.append((label, "it needs a secret"))
                continue
            script = _fill(run.strip(), matrix)
            if script is None:
                not_run.append((label, "it uses a workflow expression"))
                continue
            if _OUTSIDE.search(script) and not _SETUP.match(script):
                not_run.append((label, "it reaches outside the sandbox"))
                continue
            workdir = _text(step.get("working-directory")) or _text(
                (job.get("defaults") or {}).get("run", {}).get("working-directory")
            )
            entry = Step(label, script, safe_workdir(workdir), _text(job_id))
            (setup if _SETUP.match(script) else gates).append(entry)
    if not (setup or gates or not_run):
        return None
    return Plan(tuple(setup), tuple(gates), tuple(not_run), python, node, name or "workflow")


def safe_workdir(path: str) -> str:
    """A folder inside the checkout: ``.`` for anything absolute or climbing out."""

    path = path.replace("\\", "/").strip()
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts or path.startswith("/") or ".." in parts or ":" in path:
        return "."
    return "/".join(parts)


def _merge(plans: list[Plan], source: str) -> Plan:
    def unique(steps: list[Step]) -> tuple[Step, ...]:
        seen: dict[str, Step] = {}
        for step in steps:
            seen.setdefault(step.script, step)
        return tuple(seen.values())

    return Plan(
        setup=unique([s for p in plans for s in p.setup]),
        gates=unique([s for p in plans for s in p.gates]),
        not_run=tuple(dict.fromkeys(n for p in plans for n in p.not_run)),
        python=next((p.python for p in plans if p.python), ""),
        node=next((p.node for p in plans if p.node), ""),
        source=source,
    )


def from_workflows(repo: Path) -> Plan:
    """The plan from every pull-request workflow in the repo's checkout."""

    folder = repo / WORKFLOWS
    plans = []
    if folder.is_dir():
        for path in sorted(folder.glob("*.y*ml")):
            if path.is_symlink() or not path.is_file():
                continue
            found = from_workflow(path.read_text(encoding="utf-8", errors="replace"), path.name)
            if found is not None:
                plans.append(found)
    return _merge(plans, "the repo's CI workflows") if plans else Plan()


def from_conventions(repo: Path) -> Plan:
    """The usual checks of a project with no CI to read: what its files say it uses."""

    setup: list[Step] = []
    gates: list[Step] = []
    pyproject = repo / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8", errors="replace") if pyproject.is_file() else ""
    requirements = next(
        (
            p
            for p in ("requirements_test.txt", "requirements-dev.txt", "requirements.txt")
            if (repo / p).is_file()
        ),
        "",
    )
    if requirements:
        setup.append(Step("Install Python dependencies", f"pip install -r {requirements}"))
    if "[tool.ruff" in text:
        gates += [Step("ruff check", "ruff check ."), Step("ruff format", "ruff format --check .")]
    if (repo / "tests").is_dir() or "[tool.pytest" in text:
        gates.append(Step("pytest", "python -m pytest -q"))
    package = repo / "package.json"
    if package.is_file():
        try:
            scripts = (json.loads(package.read_text(encoding="utf-8")).get("scripts")) or {}
        except ValueError, OSError:
            scripts = {}
        if (repo / "package-lock.json").is_file():
            setup.append(Step("Install Node dependencies", "npm ci"))
        for script in ("lint", "test"):
            if isinstance(scripts, dict) and script in scripts:
                gates.append(Step(f"npm run {script}", f"npm run {script}"))
    makefile = repo / "Makefile"
    if not gates and makefile.is_file():
        targets = makefile.read_text(encoding="utf-8", errors="replace")
        for target in ("check", "test", "lint"):
            if re.search(rf"^{target}\s*:", targets, re.MULTILINE):
                gates.append(Step(f"make {target}", f"make {target}"))
    if not gates:
        return Plan()
    return Plan(tuple(setup), tuple(gates), source="the project's conventions")


def from_override(repo: Path) -> Plan | None:
    """``.watchtower/gates.toml``: ``setup = [...]`` and ``gates = [...]`` command lists."""

    path = repo / OVERRIDE
    if path.is_symlink() or not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError, OSError:
        return None

    def steps(key: str) -> tuple[Step, ...]:
        found = data.get(key)
        commands = (
            [c for c in found if isinstance(c, str) and c.strip()]
            if isinstance(found, list)
            else []
        )
        return tuple(Step(c.strip().splitlines()[0][:60], c.strip()) for c in commands[:MAX_STEPS])

    plan = Plan(
        setup=steps("setup"),
        gates=steps("gates"),
        python=str(data.get("python", "")),
        node=str(data.get("node", "")),
        source=f"the repo's {OVERRIDE}",
    )
    return plan if plan else None


def plan(repo: Path) -> Plan:
    """The checks for the checkout ``repo`` (the default branch): the override if the repo
    has one, else its CI workflows, else its conventions. Empty if none are found."""

    return from_override(repo) or (from_workflows(repo) or from_conventions(repo))
