"""Replaying closed issues, to measure how well the drafts judge them.

Each closed issue that a stranger opened and a maintainer answered is cut just
before the maintainer's first answer (labels removed, state "open"), and drafted
as if it were new: ``drafts.generate`` with ``as_of`` set to when the newest
message before that answer came, so releases and earlier threads from later
(like the release that fixed it) stay hidden. The report puts the model's assessment and reply
next to what really happened: the maintainer's first answer and how the issue
was closed. Judging that is the user's part; the report is the evidence, and a
rerun after a change shows whether it helped.

The code it may investigate is the author's version, or else the newest release
at the cutoff -- never today's default branch, which would already hold the fix.

It needs the model and the internet (attachments are downloaded without a token,
the code with the read token), so it runs in the watcher container, from a shell:
``python -m watchtower eval <owner/name> [--limit N] [number ...]``. The report,
``/data/eval/<name>-<time>.md``, is rewritten after every issue.
"""

from __future__ import annotations

import calendar
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import attachments, drafts, snapshot
from .analysis import CATEGORIES
from .config import Config
from .history import History
from .store import Draft

MAX_QUOTED = 1500


@dataclass(frozen=True)
class Case:
    number: int
    title: str
    thread: dict  # as it was just before the first maintainer answer
    newest_url: str
    # When the draft would have been written: the newest message before the answer.
    # (Not the answer's time: a maintainer often releases the fix, then answers.)
    cutoff: str
    answer: str  # the maintainer's first answer
    outcome: str  # how the issue was closed


@dataclass(frozen=True)
class Outcome:
    case: Case
    result: drafts.Result | None
    seconds: float


def _later(stamp: str) -> str:
    """An ISO time one second later (``as_of`` compares with ``<``)."""

    seconds = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")) + 1
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def cases(history: History, repo: str, numbers: Sequence[int] = ()) -> list[Case]:
    found = []
    for number in numbers or history.closed(repo):
        thread = history.thread(repo, number)
        if thread is None or thread["maintainer"]:
            continue  # unknown, or opened by a maintainer: nothing to triage
        first = next((c for c in thread["comments"] if c["maintainer"]), None)
        if first is None:
            continue
        before = [c for c in thread["comments"] if c["created"] < first["created"]]
        as_it_was = {**thread, "state": "open", "labels": "", "comments": before}
        newest = before[-1] if before else thread
        found.append(
            Case(
                number=number,
                title=thread["title"],
                thread=as_it_was,
                newest_url=newest["url"],
                # A second later: what was there when the newest message arrived.
                cutoff=_later(newest["created"]),
                answer=first["body"],
                outcome=thread["state"],
            )
        )
    return found


def _quote(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_QUOTED:
        text = text[:MAX_QUOTED].rstrip() + " …"
    return "\n".join(f"> {line}" for line in text.splitlines()) or "> (empty)"


def report(repo: str, cfg: Config, outcomes: Sequence[Outcome], started: str) -> str:
    lines = [
        f"# Replay of {repo}, {started}",
        "",
        f"{len(outcomes)} closed issue(s), each cut just before the maintainer's first answer."
        f" Model {cfg.agent_model}, context {cfg.agent_num_ctx} tokens.",
        "",
        "| # | Model says | Confidence | Evidence checked | Steps | Closed as | Minutes |",
        "|---|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        v = o.result.verdict if o.result else None
        checked = f"{sum(e.verified for e in v.evidence)}/{len(v.evidence)}" if v else "–"
        lines.append(
            f"| #{o.case.number} | {CATEGORIES[v.category] if v else 'failed'}"
            f" | {v.confidence if v else '–'} | {checked} | {len(v.looked_at) if v else '–'}"
            f" | {o.case.outcome}"
            f" | {o.seconds / 60:.1f} |"
        )
    for o in outcomes:
        lines += ["", f"## #{o.case.number} {o.case.title}", ""]
        if o.result is None:
            lines.append("**The draft failed.**")
        else:
            v = o.result.verdict
            lines.append(
                f"**Model:** {CATEGORIES[v.category]} ({v.confidence}), assessed in"
                f" {v.attempts} attempt(s); attachments: {o.result.attachments}"
            )
            for e in v.evidence:
                mark = "✓" if e.verified else "✗ not found"
                lines.append(f"- {mark} [{e.source}] «{e.quote}»: {e.point}")
            lines += [f"- Missing: {m}" for m in v.missing]
            if v.code:
                lines.append(f"- Where: {v.code}")
            if v.fix:
                lines.append(f"- Fix: {v.fix}")
            if v.unknown_paths:
                lines.append(f"- ✗ No such file in the code: {', '.join(v.unknown_paths)}")
            if v.looked_at:
                lines.append(f"- Looked at: {'; '.join(v.looked_at)}")
            if o.result.note:
                lines.append(f"- Note: {o.result.note}")
            lines += ["", "**Model's reply:**", "", _quote(o.result.reply)]
        lines += [
            "",
            f"**Your first answer** (closed as {o.case.outcome}):",
            "",
            _quote(o.case.answer),
        ]
    return "\n".join(lines) + "\n"


def run(
    cfg: Config,
    history: History,
    repo: str,
    root: Path,
    folder: Path,
    out: Path,
    numbers: Sequence[int] = (),
    limit: int | None = None,
    say: Callable[[str], None] = print,
    source: snapshot.Source | None = None,
) -> list[Outcome]:
    """Replay; with a ``source`` (GitHub), each case gets the code at its version."""

    selected = cases(history, repo, numbers)[:limit]
    started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    say(f"{len(selected)} issue(s) to replay; report: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    outcomes: list[Outcome] = []
    for case in selected:
        try:
            attachments.download(case.thread, repo, folder)
        except Exception as err:  # noqa: BLE001 -- replay without the files
            say(f"#{case.number}: attachments failed ({type(err).__name__})")
        if source is not None:
            try:
                drafts.fetch_code(source, history, repo, case.thread, folder, root, case.cutoff)
            except Exception as err:  # noqa: BLE001 -- replay without the code
                say(f"#{case.number}: code failed ({type(err).__name__})")
        draft = Draft(
            id=0,
            event_key=f"eval:{repo}#{case.number}",
            repo=repo,
            number=case.number,
            kind="issue",
            topic="triage",
            title=case.title,
            url=case.newest_url,
            reply_to="",
            status="eval",
            note="",
            reason="",
            posted_url="",
            attachments="",
            verdict="",
        )
        clock = time.monotonic()
        result = drafts.generate(
            cfg,
            draft,
            history,
            root,
            folder,
            thread=case.thread,
            as_of=case.cutoff,
            beat=lambda step, n=case.number: say(f"#{n}: {step}"),
        )
        outcomes.append(Outcome(case, result, time.monotonic() - clock))
        out.write_text(report(repo, cfg, outcomes, started), encoding="utf-8")
        verdict = result.verdict.category if result else "failed"
        say(f"#{case.number}: {verdict}, {outcomes[-1].seconds / 60:.1f} min")
    return outcomes
