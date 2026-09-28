"""Replaying closed issues, to measure how well the drafts judge them.

Each closed issue that a stranger opened and a maintainer answered is cut just
before the maintainer's first answer (labels removed, state "open"), and drafted
as if it were new: ``drafts.generate`` with ``as_of``, so releases and earlier
threads from later stay hidden. The report puts the model's assessment and reply
next to what really happened: the maintainer's first answer and how the issue
was closed. Judging that is the user's part; the report is the evidence, and a
rerun after a change shows whether it helped.

It needs the model and the internet (attachments are downloaded, without a
token), so it runs in the watcher container, from a shell:
``python -m watchtower eval <owner/name> [--limit N] [number ...]``. The report,
``/data/eval/<name>-<time>.md``, is rewritten after every issue.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import attachments, drafts
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
    cutoff: str  # when that answer came
    answer: str  # the maintainer's first answer
    outcome: str  # how the issue was closed


@dataclass(frozen=True)
class Outcome:
    case: Case
    result: drafts.Result | None
    seconds: float


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
        found.append(
            Case(
                number=number,
                title=thread["title"],
                thread=as_it_was,
                newest_url=before[-1]["url"] if before else thread["url"],
                cutoff=first["created"],
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
        "| # | Model says | Confidence | Evidence checked | Closed as | Minutes |",
        "|---|---|---|---|---|---|",
    ]
    for o in outcomes:
        v = o.result.verdict if o.result else None
        checked = f"{sum(e.verified for e in v.evidence)}/{len(v.evidence)}" if v else "–"
        lines.append(
            f"| #{o.case.number} | {CATEGORIES[v.category] if v else 'failed'}"
            f" | {v.confidence if v else '–'} | {checked} | {o.case.outcome}"
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
) -> list[Outcome]:
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
