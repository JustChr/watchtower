"""Replaying closed issues, to measure how well the drafts judge them.

Each closed issue that a stranger opened and a maintainer answered is cut just
before the maintainer's first answer (labels removed, state "open"), and drafted
as if it were new: ``drafts.generate`` with ``as_of`` set to when the newest
message before that answer came, so releases and earlier threads from later
(like the release that fixed it) stay hidden. The report puts the model's assessment and reply
next to what really happened: the maintainer's first answer and how the issue
was closed. Judging that is the user's part; the report is the evidence, and a
rerun after a change shows whether it helped.

A case can also be cut at a given comment, open issues too (``160@5``: answer the
5th comment, as it was then). The report then shows what came next: the
maintainer's next answer, or else the next comment by anyone (a bad reply that
got posted, kept as the example to beat).

The code it may investigate is the author's version, or else the newest release
at the cutoff -- never today's default branch, which would already hold the fix.

It needs the internet (attachments are downloaded without a token, the code with
the read token) and the model, which no container has both of. So from a shell in
the watcher container, ``python -m watchtower eval <owner/name> [--limit N]
[number[@comment] ...]`` fetches the files and the code (``prepare``) and queues
the replay for the worker; Telegram says when it's done. The report,
``/data/eval/<name>-<time>.md``, is rewritten after every issue. From the dev
machine, ``run`` with a ``source`` does both.
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
    outcome: str  # how the issue was closed ("open" if it isn't)
    answer_label: str = "Your first answer"


@dataclass(frozen=True)
class Outcome:
    case: Case
    result: drafts.Result | None
    seconds: float


def _later(stamp: str) -> str:
    """An ISO time one second later (``as_of`` compares with ``<``)."""

    seconds = calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ")) + 1
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(seconds))


def case_at(history: History, repo: str, number: int, at: int) -> Case | None:
    """``number`` cut just after its ``at``-th comment (from 1), which is to be
    answered; ``None`` if there's no such comment or a maintainer wrote it."""

    thread = history.thread(repo, number)
    if thread is None or not 1 <= at <= len(thread["comments"]):
        return None
    newest = thread["comments"][at - 1]
    if newest["maintainer"]:
        return None
    later = thread["comments"][at:]
    answer = next((c for c in later if c["maintainer"]), later[0] if later else None)
    if answer is None:
        label = "Nothing came next"
    elif answer["maintainer"]:
        label = "Your next answer"
    else:
        label = f"What came next, by {answer['author']}"
    return Case(
        number=number,
        title=thread["title"],
        thread={**thread, "state": "open", "labels": "", "comments": thread["comments"][:at]},
        newest_url=newest["url"],
        cutoff=_later(newest["created"]),
        answer=answer["body"] if answer else "",
        outcome=thread["state"],
        answer_label=label,
    )


def cases(history: History, repo: str, numbers: Sequence[int | tuple[int, int]] = ()) -> list[Case]:
    """The cases to replay: ``numbers`` (``(number, comment)`` cuts at that comment),
    else every closed issue."""

    found = []
    for number in numbers or history.closed(repo):
        if isinstance(number, tuple):
            case = case_at(history, repo, *number)
            found += [case] if case else []
            continue
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
        f"{len(outcomes)} issue(s), each cut just before the maintainer's first answer"
        " (or at the comment given)."
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
            for a in v.asks:
                lines.append(f"- Asked: «{a.quote}»" + ("" if a.verified else " ✗ not found"))
            if v.decision:
                lines.append(f"- Yours to decide: {v.decision}")
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
            if o.result.seconds:
                passes = " · ".join(f"{k} {s / 60:.1f}" for k, s in o.result.seconds.items())
                lines.append(f"- Minutes per pass: {passes}")
            lines += ["", "**Model's reply:**", "", _quote(o.result.reply)]
        state = "still open" if o.case.outcome == "open" else f"closed as {o.case.outcome}"
        lines += ["", f"**{o.case.answer_label}** ({state}):", "", _quote(o.case.answer)]
    return "\n".join(lines) + "\n"


def prepare(
    history: History,
    repo: str,
    selected: Sequence[Case],
    root: Path,
    folder: Path,
    source: snapshot.Source,
    say: Callable[[str], None] = print,
) -> None:
    """Download each case's attachments and fetch the code at its version."""

    for case in selected:
        try:
            attachments.download(case.thread, repo, folder)
        except Exception as err:  # noqa: BLE001 -- replay without the files
            say(f"#{case.number}: attachments failed ({type(err).__name__})")
        try:
            drafts.fetch_code(source, history, repo, case.thread, folder, root, case.cutoff)
        except Exception as err:  # noqa: BLE001 -- replay without the code
            say(f"#{case.number}: code failed ({type(err).__name__})")


def run(
    cfg: Config,
    history: History,
    repo: str,
    root: Path,
    folder: Path,
    out: Path,
    numbers: Sequence[int | tuple[int, int]] = (),
    limit: int | None = None,
    say: Callable[[str], None] = print,
    source: snapshot.Source | None = None,
) -> list[Outcome]:
    """Replay; with a ``source`` (GitHub) it first fetches each case's files and
    code (``prepare``), without one it uses what was fetched before."""

    selected = cases(history, repo, numbers)[:limit]
    started = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    say(f"{len(selected)} issue(s) to replay; report: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        prepare(history, repo, selected, root, folder, source, say)
    outcomes: list[Outcome] = []
    for case in selected:
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
        try:
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
        except drafts.Unfit as err:
            say(f"#{case.number}: not drafted, {err}")
            result = None
        outcomes.append(Outcome(case, result, time.monotonic() - clock))
        out.write_text(report(repo, cfg, outcomes, started), encoding="utf-8")
        verdict = result.verdict.category if result else "failed"
        say(f"#{case.number}: {verdict}, {outcomes[-1].seconds / 60:.1f} min")
    return outcomes
