"""A pull request as the worker reviews it: what the watcher fetches for it.

The worker has no internet, so the watcher (read-only token) puts a PR's facts, its
per-file patches and the code at its head commit under
``<repos>/.prs/<owner>/<name>/<number>/`` (``pr.json`` and ``code/``). The strangers'
text in it (title, body, patches, the code) is data, and the code is never run here.

A fork's commit is fetched from the base repo by its id: GitHub keeps every PR's
commits there. The copy is refreshed when the head commit changes, and only the
``KEEP`` most recently used stay.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from . import snapshot

PRS = ".prs"  # under the repos root
KEEP = 10  # PR copies kept per repo
MAX_FILES = 300  # changed files listed (GitHub lists up to 3000)
MAX_PATCH = 20_000  # characters of one file's patch kept
PAGE = 100
_SHA = re.compile(r"[0-9a-f]{40}")


class Source(Protocol):
    def get_json(self, path: str) -> Any: ...
    def download(self, path: str, dest: Path, max_bytes: int) -> None: ...


def folder(root: Path, repo: str, number: int) -> Path:
    return root.joinpath(PRS, *repo.split("/"), str(int(number)))


def code_path(root: Path, repo: str, number: int) -> Path:
    return folder(root, repo, number) / "code"


def load(root: Path, repo: str, number: int) -> dict | None:
    """The PR facts the watcher fetched, or ``None`` if it hasn't."""

    try:
        data = json.loads((folder(root, repo, number) / "pr.json").read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return data if isinstance(data, dict) else None


def _user(node: dict | None) -> str:
    return (node or {}).get("login") or "ghost"


def facts(pull: dict, files: list[dict]) -> dict:
    """The parts of GitHub's answers a review needs, in one JSON-able dict."""

    head, base = pull.get("head") or {}, pull.get("base") or {}
    return {
        "number": pull["number"],
        "title": pull.get("title") or "",
        "author": _user(pull.get("user")),
        "association": pull.get("author_association") or "NONE",
        "body": pull.get("body") or "",
        "state": pull.get("state") or "",
        "draft": bool(pull.get("draft")),
        "url": pull.get("html_url") or "",
        "base_ref": base.get("ref") or "",
        "base_sha": base.get("sha") or "",
        "head_sha": head.get("sha") or "",
        "head_repo": ((head.get("repo") or {}).get("full_name")) or "",
        "head_ref": head.get("ref") or "",
        "maintainer_can_modify": bool(pull.get("maintainer_can_modify")),
        "mergeable": pull.get("mergeable"),
        "commits": pull.get("commits") or 0,
        "additions": pull.get("additions") or 0,
        "deletions": pull.get("deletions") or 0,
        "changed_files": pull.get("changed_files") or len(files),
        "labels": [label.get("name", "") for label in pull.get("labels") or []],
        "files": [
            {
                "path": f.get("filename") or "",
                "status": f.get("status") or "",
                "previous": f.get("previous_filename") or "",
                "additions": f.get("additions") or 0,
                "deletions": f.get("deletions") or 0,
                "patch": (f.get("patch") or "")[:MAX_PATCH],
                "patch_cut": len(f.get("patch") or "") > MAX_PATCH,
            }
            for f in files[:MAX_FILES]
        ],
    }


def _files(source: Source, repo: str, number: int) -> list[dict]:
    found: list[dict] = []
    for page in range(1, MAX_FILES // PAGE + 2):
        got = source.get_json(f"/repos/{repo}/pulls/{number}/files?per_page={PAGE}&page={page}")
        found += got
        if len(got) < PAGE or len(found) >= MAX_FILES:
            break
    return found


def fetch(source: Source, repo: str, number: int, root: Path) -> dict:
    """Bring the PR's copy up to date: the facts and files every time, the code when
    the head commit changed. Returns the facts. A head whose code can't be fetched
    still gets its facts (the review says it saw only the diff)."""

    pull = source.get_json(f"/repos/{repo}/pulls/{number}")
    data = facts(pull, _files(source, repo, number))
    where = folder(root, repo, number)
    sha = data["head_sha"]
    if not _SHA.fullmatch(sha):
        raise snapshot.SnapshotError("unexpected commit id")
    old = load(root, repo, number)
    code = where / "code"
    data["code"] = bool(old and old.get("head_sha") == sha and old.get("code") and code.is_dir())
    where.mkdir(parents=True, exist_ok=True)
    if not data["code"]:
        shutil.rmtree(code, ignore_errors=True)
        try:
            snapshot.fetch(source, repo, sha, code)
            data["code"] = True
        except Exception:  # noqa: BLE001 -- the review works from the patches alone
            data["code"] = False
    tmp = where / "pr.json.new"
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    tmp.replace(where / "pr.json")
    os.utime(where)
    prune(root, repo)
    return data


def prune(root: Path, repo: str) -> None:
    """Keep the ``KEEP`` PR copies of ``repo`` used last."""

    parent = root.joinpath(PRS, *repo.split("/"))
    if not parent.is_dir():
        return
    copies = [p for p in parent.iterdir() if p.is_dir() and p.name.isdigit()]
    for old in sorted(copies, key=lambda p: p.stat().st_mtime, reverse=True)[KEEP:]:
        shutil.rmtree(old, ignore_errors=True)


MERGE_TRIES = 4  # GitHub works out ``mergeable`` lazily: null at first
MERGE_WAIT = 3.0  # seconds between asks


def fetch_merged(
    source: Source,
    repo: str,
    number: int,
    root: Path,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Path | None:
    """The PR merged into its base, the tree the checks run on (what CI tests). ``None``
    if it doesn't merge -- the review then asks for a rebase and no check runs -- or if
    GitHub hasn't said after a few asks. The merge commit is pinned by its id: the merge
    ref moves whenever the base does. Kept while that commit is the PR's current one."""

    pull: dict = {}
    for attempt in range(MERGE_TRIES):
        pull = source.get_json(f"/repos/{repo}/pulls/{number}")
        if pull.get("mergeable") is not None:
            break
        if attempt < MERGE_TRIES - 1:
            sleep(MERGE_WAIT)
    sha = pull.get("merge_commit_sha") or ""
    if pull.get("mergeable") is not True or not _SHA.fullmatch(sha):
        return None
    where = folder(root, repo, number)
    target, marker = where / "merged", where / "merged.sha"
    if target.is_dir() and marker.is_file() and marker.read_text(encoding="utf-8") == sha:
        return target
    where.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(target, ignore_errors=True)
    snapshot.fetch(source, repo, sha, target)
    marker.write_text(sha, encoding="utf-8")
    os.utime(where)
    return target
