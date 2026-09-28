"""Files attached to issues and comments (diagnostics, logs), for the drafts.

The watcher downloads them when it queues a draft; the drafter, which has no
internet, reads them from ``/data/attachments/<owner>/<name>/``. Only GitHub's
own upload links count (``github.com/user-attachments/files/<id>/<name>`` and
the older ``github.com/<owner>/<repo>/files/<id>/<name>``), fetched without a
token (they're public for public repos), up to ``MAX_BYTES``, and kept only if
they are text. They are strangers' data like any issue text: never executed,
never unpacked, only put in a prompt.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

MAX_BYTES = 2 * 1024 * 1024
MAX_FILENAME = 80
TIMEOUT = 30

_FILE = re.compile(
    r"https://github\.com/(?:user-attachments|[\w.-]+/[\w.-]+)/files/(\d+)/([^\s)\]<>\"']+)",
    re.IGNORECASE,
)
_ASSET = re.compile(r"https://github\.com/user-attachments/assets/[\w-]+", re.IGNORECASE)
_UNSAFE = re.compile(r"[^\w.-]+")


@dataclass(frozen=True)
class Link:
    url: str
    file_id: str
    name: str  # cleaned: letters, digits, ``._-`` only
    author: str


def clean_name(raw: str) -> str:
    """A stranger's file name, reduced to letters, digits, ``._-``: it goes in
    prompts and paths, so it mustn't carry words or path separators."""

    return _UNSAFE.sub("_", urllib.parse.unquote(raw)).strip(".")[:MAX_FILENAME] or "file"


def links(thread: dict) -> list[Link]:
    """Every uploaded file in the thread, once, in order (opening post first)."""

    found: dict[str, Link] = {}
    for entry in (thread, *thread["comments"]):
        for match in _FILE.finditer(entry["body"]):
            file_id, name = match[1], clean_name(match[2])
            found.setdefault(file_id, Link(match[0], file_id, name, entry["author"]))
    return list(found.values())


def images(thread: dict) -> int:
    """How many images or videos the thread has (not readable for the model)."""

    return sum(len(set(_ASSET.findall(e["body"]))) for e in (thread, *thread["comments"]))


def path_for(folder: Path, repo: str, link: Link) -> Path:
    return folder.joinpath(*repo.split("/"), f"{link.file_id}-{link.name}")


def _text(data: bytes) -> str | None:
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def download(
    thread: dict,
    repo: str,
    folder: Path,
    opener: Callable[..., object] | None = None,
) -> int:
    """Fetch the thread's files not fetched yet; returns how many were saved.

    A file that fails, is too big or isn't text is skipped (the drafter says so).
    No token is sent: not to github.com, and not along the redirect to storage.
    """

    opener = opener or urllib.request.urlopen
    saved = 0
    for link in links(thread):
        target = path_for(folder, repo, link)
        if target.exists():
            continue
        request = urllib.request.Request(link.url, headers={"User-Agent": "watchtower"})
        try:
            with opener(request, timeout=TIMEOUT) as response:
                data = response.read(MAX_BYTES + 1)
        except (urllib.error.URLError, OSError, ValueError) as err:
            _LOGGER.warning("attachment %s: %s", link.file_id, type(err).__name__)
            continue
        if len(data) > MAX_BYTES or _text(data) is None:
            _LOGGER.info("attachment %s skipped: too big or not text", link.file_id)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(data)
        partial.replace(target)
        saved += 1
    return saved


def read(folder: Path, repo: str, link: Link) -> str | None:
    """The saved file as prompt text (JSON minified), or ``None`` if there isn't one."""

    target = path_for(folder, repo, link)
    if target.is_symlink() or not target.is_file():
        return None
    text = _text(target.read_bytes())
    if text is None:
        return None
    try:
        return json.dumps(json.loads(text), ensure_ascii=False, separators=(",", ":"))
    except ValueError:
        return text


def fit(text: str, limit: int) -> str:
    """``text`` within ``limit``: its start and its end (a log's latest lines), the middle cut."""

    if len(text) <= limit:
        return text
    marker = f"\n[... {len(text) - limit} characters left out ...]\n"
    head = max(0, (limit - len(marker)) // 4)
    tail = max(0, limit - len(marker) - head)
    return text[:head] + marker + (text[-tail:] if tail else "")
