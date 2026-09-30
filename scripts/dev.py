#!/usr/bin/env python3
"""Dev tool: the steps of a working session, so nobody re-invents them.

    python scripts/dev.py check [pytest args]   ruff format + ruff check + pytest, short output
    python scripts/dev.py sub SPEC.json|SPEC.py batch exact replacements; a .py spec defines
                                                EDITS = [{"file", "old", "new"}] in r'''...'''
    python scripts/dev.py ci [--wait]           CI of HEAD; --wait polls, prints why it failed
    python scripts/dev.py ship MSGFILE [--no-push] [--trailer TEXT]
                                                check, commit everything, push, wait for CI
    python scripts/dev.py clean                 delete caches (OneDrive syncs them)
    python scripts/dev.py status                git state, unpushed commits, last CI run

Why it exists: batch edits through a shell heredoc turn ``\\n`` in Python strings into real
newlines and silently miss (hit again and again), a long ``sleep`` in one call times out,
and the same ruff/pytest/gh sequences were retyped every session. Write the spec or the
message with the Write tool (no shell escaping), then run this.

Standard library only. ``ship`` pushes: run it only when the user said to push.
"""

from __future__ import annotations

import json
import re
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAILER = "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
CACHES = (".pytest_cache", ".ruff_cache", "__pycache__")
CI_POLL = 10  # seconds between asks of GitHub
CI_LIMIT = 900  # seconds --wait gives a run
_LOG_LINES = re.compile(r"Error|FAILED|^E |assert|Traceback", re.IGNORECASE)


def run(*cmd: str, quiet: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, cwd=ROOT, capture_output=quiet, text=True, encoding="utf-8", errors="replace"
    )


def out(*cmd: str) -> str:
    return run(*cmd, quiet=True).stdout.strip()


# -- check ----------------------------------------------------------------------------------


def check(pytest_args: list[str]) -> int:
    """Format (rewrites files), lint, test. Prints the last lines of each, stops on none."""

    failed = []
    for name, cmd in (
        ("ruff format", [sys.executable, "-m", "ruff", "format", "."]),
        ("ruff check", [sys.executable, "-m", "ruff", "check", "."]),
        ("pytest", [sys.executable, "-m", "pytest", "-q", *pytest_args]),
    ):
        done = run(*cmd, quiet=True)
        text = (done.stdout + done.stderr).strip().splitlines()
        shown = text if done.returncode else text[-1:]
        print(f"[{'ok' if done.returncode == 0 else 'FAILED'}] {name}: " + "\n".join(shown[-40:]))
        if done.returncode:
            failed.append(name)
    return 1 if failed else 0


# -- sub ------------------------------------------------------------------------------------


def apply_edits(edits: list[dict], base: Path = ROOT) -> list[str]:
    """Apply ``[{"file", "old", "new", "all"?}]``: every ``old`` must be found exactly once
    (or with ``"all": true``, at least once) or nothing at all is written. Line endings
    follow each file (CRLF stays CRLF). Returns what was done, one line per edit."""

    contents: dict[Path, str] = {}
    crlf: dict[Path, bool] = {}
    report = []
    for number, edit in enumerate(edits, 1):
        path = base / edit["file"]
        if path not in contents:
            raw = path.read_bytes().decode("utf-8")
            crlf[path] = "\r\n" in raw
            contents[path] = raw.replace("\r\n", "\n")
        old, new = edit["old"].replace("\r\n", "\n"), edit["new"].replace("\r\n", "\n")
        found = contents[path].count(old)
        if found == 0 or (found > 1 and not edit.get("all")):
            first = old.strip().splitlines()[0][:70] if old.strip() else "(empty)"
            raise ValueError(f"edit {number}: {found} match(es) in {edit['file']} for: {first}")
        contents[path] = contents[path].replace(old, new)
        report.append(f"edit {number}: {edit['file']} ({found} replaced)")
    for path, text in contents.items():
        path.write_bytes((text.replace("\n", "\r\n") if crlf[path] else text).encode("utf-8"))
    return report


def load_spec(spec: str) -> list[dict]:
    """The edits of a spec file: JSON, or a ``.py`` file defining ``EDITS`` (a list of the
    same dicts) -- raw triple-quoted strings there need no escaping for code."""

    if spec.endswith(".py"):
        return runpy.run_path(spec)["EDITS"]
    return json.loads(Path(spec).read_text(encoding="utf-8"))


def sub(spec: str) -> int:
    try:
        edits = load_spec(spec)
        print("\n".join(apply_edits(edits)))
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as err:
        print(f"nothing written: {err}", file=sys.stderr)
        return 1
    return 0


# -- ci -------------------------------------------------------------------------------------


def latest_run(sha: str) -> dict | None:
    text = out(
        "gh",
        "run",
        "list",
        "--commit",
        sha,
        "--limit",
        "1",
        "--json",
        "databaseId,status,conclusion,displayTitle",
    )
    found = json.loads(text) if text else []
    return found[0] if found else None


def ci(wait: bool) -> int:
    """The CI run of HEAD. With ``wait``, until it completes; a failure prints its log."""

    sha = out("git", "rev-parse", "HEAD")
    started = time.monotonic()
    while True:
        found = latest_run(sha)
        if found and found["status"] == "completed":
            break
        if not wait or time.monotonic() - started > CI_LIMIT:
            print(f"CI of {sha[:7]}: {found['status'] if found else 'no run yet'}")
            return 0 if not wait else 1
        time.sleep(CI_POLL)
    print(f"CI of {sha[:7]}: {found['conclusion']} ({found['displayTitle']})")
    if found["conclusion"] == "success":
        return 0
    log = run("gh", "run", "view", str(found["databaseId"]), "--log-failed", quiet=True).stdout
    lines = [line.split("\t", 2)[-1][:200] for line in log.splitlines() if _LOG_LINES.search(line)]
    print("\n".join(lines[-30:]))
    return 1


# -- ship, clean, status ---------------------------------------------------------------------


def ship(message_file: str, push: bool, trailer: str) -> int:
    message = Path(message_file).read_text(encoding="utf-8").strip()
    if not message:
        print("empty commit message", file=sys.stderr)
        return 2
    if check([]):
        return 1
    run("git", "add", "-A")
    message_path = Path(message_file).resolve()
    if message_path.is_relative_to(ROOT):  # the message file isn't part of the commit
        run("git", "reset", "-q", "--", str(message_path.relative_to(ROOT)), quiet=True)
    if not out("git", "diff", "--cached", "--name-only"):
        print("nothing to commit")
        return 0
    done = run("git", "commit", "-q", "-m", f"{message}\n\n{trailer}", quiet=True)
    if done.returncode:
        print(done.stdout + done.stderr, file=sys.stderr)
        return 1
    print(out("git", "log", "--oneline", "-1"))
    if not push:
        return 0
    pushed = run("git", "push", "origin", "HEAD", quiet=True)
    print((pushed.stdout + pushed.stderr).strip().splitlines()[-1:] or "pushed")
    return ci(wait=True) if pushed.returncode == 0 else 1


def clean() -> int:
    """Delete the caches, wherever they are (literal paths: no ``rm`` safety check trips)."""

    removed = 0
    for name in CACHES:
        for path in ROOT.rglob(name):
            if ".git" not in path.parts and path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1
    print(f"removed {removed} cache folder(s)")
    return 0


def status() -> int:
    print(out("git", "status", "-sb"))
    ahead = out("git", "log", "--oneline", "@{u}..HEAD")
    print("unpushed:\n" + ahead if ahead else "nothing unpushed")
    return ci(wait=False)


def main(argv: list[str]) -> int:
    match argv:
        case ["check", *rest]:
            return check(rest)
        case ["sub", spec]:
            return sub(spec)
        case ["ci"]:
            return ci(False)
        case ["ci", "--wait"]:
            return ci(True)
        case ["ship", message, *flags]:
            trailer = TRAILER
            if "--trailer" in flags:
                trailer = flags[flags.index("--trailer") + 1]
            return ship(message, "--no-push" not in flags, trailer)
        case ["clean"]:
            return clean()
        case ["status"]:
            return status()
        case _:
            print(__doc__)
            return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
