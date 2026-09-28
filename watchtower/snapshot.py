"""A read-only copy of each watched repo's default branch, and the docs to learn it from.

The watcher downloads the branch as a tarball (no ``git`` in the image) into
``/data/repos/<owner>/<name>``, only when the head commit changed. Workers will
get that folder read-only instead of cloning.

The tarball is outside input: it is unpacked with tarfile's ``data`` filter (no
paths outside the target, no device files, no links pointing out), after a
size and file-count check, into a staging folder that then replaces the old copy.

The default branch holds what the maintainers merged, so its docs
(``trusted_docs``) are the most trustworthy text about a repo there is. A
release's tag can be fetched the same way (``fetch``) for the repo brief.
"""

from __future__ import annotations

import os
import re
import shutil
import tarfile
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .history import History
from .llm import scrub

MAX_DOWNLOAD = 200 * 1024 * 1024
MAX_UNPACKED = 500 * 1024 * 1024
MAX_FILES = 50_000
MAX_DOC = 20_000
DOCS_BUDGET = 60_000
TREE_LIMIT = 400
SKIP_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"})
# Top-level docs, most useful first; matched case-insensitively.
TOP_DOCS = (
    "claude.md",
    "agents.md",
    "readme.md",
    "readme.rst",
    "readme.txt",
    "readme",
    "contributing.md",
)

_SHA = re.compile(r"[0-9a-f]{40}")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"<[a-zA-Z/][^>]*>")
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


class SnapshotError(Exception):
    pass


class Source(Protocol):
    def get_json(self, path: str) -> Any: ...
    def download(self, path: str, dest: Path, max_bytes: int) -> None: ...


def path_for(root: Path, repo: str) -> Path:
    return root.joinpath(*repo.split("/"))


def head_commit(source: Source, repo: str) -> str:
    branch = source.get_json(f"/repos/{repo}")["default_branch"]
    data = source.get_json(f"/repos/{repo}/branches/{urllib.parse.quote(branch, safe='/')}")
    sha = data["commit"]["sha"]
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise SnapshotError("unexpected commit id")
    return sha


def unpack(archive: Path, target: Path) -> None:
    """Replace ``target`` with the archive's single top-level folder."""

    staging = target.with_name(target.name + ".new")
    old = target.with_name(target.name + ".old")
    for leftover in (staging, old):
        shutil.rmtree(leftover, ignore_errors=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        if len(members) > MAX_FILES or sum(m.size for m in members) > MAX_UNPACKED:
            raise SnapshotError("archive too large")
        try:
            tar.extractall(staging, members=members, filter="data")
        except tarfile.FilterError as err:
            shutil.rmtree(staging, ignore_errors=True)
            raise SnapshotError(f"refused archive member: {type(err).__name__}") from None
    tops = list(staging.iterdir()) if staging.exists() else []
    if len(tops) != 1 or not tops[0].is_dir():
        shutil.rmtree(staging, ignore_errors=True)
        raise SnapshotError("expected one top-level folder")
    if target.exists():
        target.rename(old)
    tops[0].rename(target)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)


def fetch(source: Source, repo: str, ref: str, target: Path) -> None:
    """Make ``target`` a copy of ``ref`` (a commit id or a tag)."""

    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.with_name(target.name + ".tar.gz")
    try:
        ref = urllib.parse.quote(ref, safe="")
        source.download(f"/repos/{repo}/tarball/{ref}", archive, MAX_DOWNLOAD)
        unpack(archive, target)
    finally:
        archive.unlink(missing_ok=True)


def sync(source: Source, history: History, repo: str, root: Path) -> str:
    """Make ``path_for(root, repo)`` the head of the default branch. Returns its commit id."""

    commit = head_commit(source, repo)
    target = path_for(root, repo)
    name = f"snapshot:{repo}"
    if history.get_cursor(name) == commit and target.is_dir():
        return commit
    fetch(source, repo, commit, target)
    history.set_cursor(name, commit)
    return commit


@dataclass(frozen=True)
class Release:
    tag: str
    prerelease: bool
    published: str
    notes: str

    @property
    def label(self) -> str:
        return f"{self.tag} ({'beta' if self.prerelease else 'stable'})"


def releases(source: Source, repo: str, limit: int = 20) -> list[Release]:
    """Published releases, newest first (drafts left out)."""

    found = [
        Release(
            tag=item["tag_name"],
            prerelease=bool(item.get("prerelease")),
            published=item.get("published_at") or "",
            notes=(item.get("body") or "")[:MAX_DOC],
        )
        for item in source.get_json(f"/repos/{repo}/releases?per_page={limit}") or []
        if not item.get("draft") and item.get("tag_name")
    ]
    return sorted(found, key=lambda r: r.published, reverse=True)


# -- reading the copy ----------------------------------------------------------


def _read(path: Path, limit: int) -> str | None:
    if path.is_symlink() or not path.is_file():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read(limit)


def trusted_docs(target: Path, budget: int = DOCS_BUDGET) -> list[tuple[str, str]]:
    """``(path, text)`` of the repo's own docs, most useful first, within ``budget`` characters:
    CLAUDE.md/AGENTS.md, README, CONTRIBUTING, ``.claude/skills/*/SKILL.md``, ``docs/**/*.md``."""

    if not target.is_dir():
        return []
    top = {p.name.lower(): p for p in target.iterdir()}
    candidates = [top[name] for name in TOP_DOCS if name in top]
    candidates += sorted((target / ".claude" / "skills").glob("*/SKILL.md"))
    candidates += sorted((target / "docs").rglob("*.md"))
    docs = []
    for path in candidates:
        if budget <= 0:
            break
        text = _read(path, min(MAX_DOC, budget))
        if text:
            docs.append((path.relative_to(target).as_posix(), text))
            budget -= len(text)
    return docs


def skill(target: Path, name: str, limit: int = MAX_DOC) -> str:
    """The repo's own ``.claude/skills/<name>/SKILL.md``, or ``""``."""

    return _read(target / ".claude" / "skills" / name / "SKILL.md", limit) or ""


def issue_templates(target: Path, budget: int = 8000) -> list[tuple[str, str]]:
    """``(path, text)`` of the repo's issue forms and templates: what reporters are asked for."""

    folder = target / ".github" / "ISSUE_TEMPLATE"
    if not folder.is_dir():
        return []
    templates = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".yml", ".yaml", ".md"} or path.stem.lower() == "config":
            continue
        text = _read(path, min(MAX_DOC, budget)) if budget > 0 else None
        if text:
            templates.append((path.relative_to(target).as_posix(), text))
            budget -= len(text)
    return templates


def tree(target: Path, limit: int = TREE_LIMIT) -> list[str]:
    """Relative file paths, sorted, without dependency and build folders."""

    paths = []
    for folder, dirs, files in os.walk(target):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        base = Path(folder).relative_to(target)
        paths += [(base / f).as_posix() for f in files]
    return sorted(paths)[:limit]


def readme_excerpt(target: Path, limit: int = 1500) -> str:
    """The README's opening as plain prose: no HTML, badges, link targets or @mentions."""

    for name, text in trusted_docs(target):
        if name.lower().startswith("readme"):
            text = _MD_LINK.sub(r"\1", _MD_IMAGE.sub("", _HTML_COMMENT.sub("", text)))
            text = " ".join(scrub(_HTML_TAG.sub(" ", text)).split())
            return text[: limit - 1].rstrip() + "…" if len(text) > limit else text
    return ""
