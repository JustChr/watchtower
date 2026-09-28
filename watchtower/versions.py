"""Which version the author runs, and what was released since -- checked by code.

Two sources: the diagnostics file Home Assistant writes (``integration_manifest``
→ ``version``: machine-written, the more reliable one) and the issue form's
"... version" field (typed by the author). When they differ, that is itself a
finding: the author may have updated after taking the diagnostics, or typed
the wrong version.

Versions come from strangers and go into the trusted "Checked by Watchtower"
section, so they are reduced to a plain version string.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .history import Release

MAX_NOTES = 6000

_VERSION = r"v?([0-9][0-9A-Za-z.+-]{0,30})"
_DIAGNOSTICS = re.compile(
    r'"integration_manifest"\s*:\s*\{[^{}]*?"version"\s*:\s*"' + _VERSION + '"'
)
# An issue-form heading about the version, then the answer on the next text line.
_FORM = re.compile(r"^#{1,6}[^\n]*\bversion\b[^\n]*\n+\s*" + _VERSION + r"\s*$", re.I | re.M)


@dataclass(frozen=True)
class Reported:
    version: str
    source: str  # "the diagnostics <file>", "the issue form"


def _norm(version: str) -> str:
    return version.lower().removeprefix("v")


def reported(thread_body: str, files: dict[str, str]) -> list[Reported]:
    """What the author runs, per source; ``files`` maps a file name to its text.

    Only the issue form's first "version" field counts: it's the project's own,
    as forms usually ask for it before Home Assistant's."""

    found = []
    for name, text in files.items():
        if match := _DIAGNOSTICS.search(text):
            found.append(Reported(match[1], f"the diagnostics {name}"))
    if match := _FORM.search(thread_body):
        found.append(Reported(match[1], "the issue form"))
    return found


def _release_of(version: str, releases: Sequence[Release]) -> Release | None:
    return next((r for r in releases if _norm(r.tag) == _norm(version)), None)


def facts(
    found: Sequence[Reported], releases: Sequence[Release], as_of: str | None = None
) -> tuple[list[str], str]:
    """``(lines, notes)``: what code can say about versions, and the release notes of
    everything newer than the author's version. ``as_of`` (ISO time) ignores later
    releases, for replaying old issues."""

    releases = [r for r in releases if as_of is None or r.published < as_of]
    lines = []
    stable = next((r for r in releases if not r.prerelease), None)
    beta = next((r for r in releases if r.prerelease), None)
    if stable is not None:
        current = f"Current stable release: {stable.tag}"
        if beta is not None and beta.published > stable.published:
            current += f"; newer beta: {beta.tag}"
        lines.append(current + ".")
    for item in found:
        lines.append(f"The author runs {item.version}, according to {item.source}.")
    if len({_norm(item.version) for item in found}) > 1:
        lines.append("These versions differ: ask which one is really installed.")
    if not found:
        return lines, ""
    version = found[0].version  # the diagnostics first
    release = _release_of(version, releases)
    if release is None:
        if releases:
            lines.append(f"{version} is not one of the published releases.")
        return lines, ""
    newer = [r for r in releases if r.published > release.published]
    if not newer:
        lines.append(f"{version} is the newest release.")
        return lines, ""
    lines.append(
        f"Released since {version}: {', '.join(r.tag for r in reversed(newer))}"
        " (release notes below)."
    )
    return lines, _notes(reversed(newer))


def _notes(releases: Iterable[Release]) -> str:
    parts, size = [], 0
    for release in releases:
        text = f"--- {release.label}\n{release.notes.strip() or '(no notes)'}"
        if size + len(text) > MAX_NOTES:
            parts.append("(later release notes left out)")
            break
        parts.append(text)
        size += len(text)
    return "\n\n".join(parts)
