"""Telegram message text (HTML parse mode). Every value from GitHub or the model is escaped."""

from __future__ import annotations

import re
from html import escape

from .analysis import CATEGORIES, Verdict
from .events import Event
from .llm import Summary
from .store import Draft, Version

# A reply to a draft starting with this is the user's own text (``drafts.own_text``).
OWN_TEXT = "text:"

HEADINGS = {
    "issue": "🆕 Issue",
    "pr": "🔀 Pull request",
    "issue_comment": "💬 Comment on issue",
    "pr_comment": "💬 Comment on PR",
    "discussion": "🗣 Discussion",
    "discussion_comment": "💬 Reply in discussion",
}
KIND_LABELS = {
    "bug": "🐞 bug",
    "feature": "✨ feature",
    "question": "❓ question",
    "support": "🛟 support",
    "docs": "📖 docs",
    "other": "📌 other",
}
# A pull request's review (``reviews``): the model's recommendation and findings.
RECOMMENDATIONS = {
    "merge": "✅ looks mergeable",
    "changes": "✏️ needs changes",
    "close": "🚫 shouldn't be merged",
    "unsure": "❔ can't tell",
}
SEVERITIES = {"blocker": "🛑", "should_fix": "⚠️", "nit": "💬"}
SNIPPET = 280
# Room for a review's findings in its assessment message, escaped.
REVIEW_ROOM = 3000

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"<[a-zA-Z/][^>]*>")


def snippet(body: str) -> str:
    """The start of a body, without issue-template comments or HTML tags, on one line."""

    text = " ".join(_HTML_TAG.sub(" ", _HTML_COMMENT.sub("", body)).split())
    if len(text) > SNIPPET:
        text = text[: SNIPPET - 1].rstrip() + "…"
    return text


def message(event: Event, summary: Summary | None, drafting: bool = False) -> str:
    repo = event.repo.split("/", 1)[1]
    lines = [
        f"<b>{HEADINGS[event.kind]} #{event.number}</b> · {escape(repo)}",
        escape(event.title),
        f"by {escape(event.author)}",
    ]
    if summary is not None:
        label = KIND_LABELS[summary.kind]
        if drafting:
            label += " · ✍️ drafting a reply…"
        elif summary.needs_reply:
            label += " · ✍️ needs a reply"
        lines += [label, f"🤖 <i>{escape(summary.text)}</i>"]
    elif text := snippet(event.body):
        lines.append(f"<blockquote>{escape(text)}</blockquote>")
    return "\n".join(lines)


def system(text: str) -> str:
    return f"⚙️ {escape(text)}"


def fit(text: str, limit: int) -> str:
    """``text`` escaped, shortened so the escaped form stays within ``limit`` characters."""

    escaped = escape(text)
    if len(escaped) <= limit:
        return escaped
    pieces, size = [], 0
    for char in text:
        piece = escape(char)
        if size + len(piece) > limit - 1:
            break
        pieces.append(piece)
        size += len(piece)
    return "".join(pieces) + "…"


def draft(d: Draft, version: Version, error: str = "", lead: str = "", label: str = "") -> str:
    """A draft up for approval. The text is shown in full -- it is exactly what gets
    posted -- so callers keep it within ``drafts.MAX_SHOWN``. ``label``: what ✅ Post
    also adds to the thread, said so the approval covers it."""

    repo = d.repo.split("/", 1)[1]
    what = "review" if d.kind == "pr" else "reply"
    head = f"<b>✍️ Draft {what} #{d.number}</b> · {escape(repo)}"
    if version.author == "user":
        head += f" · v{version.number}, your edit"
    elif version.author == "revised":
        head += f" · v{version.number}, revised as you asked"
    lines = [head, fit(d.title, 200)]
    if lead:
        lines.append(f"<b>{fit(lead, 300)}</b>")
    if error:
        lines.append(f"⚠️ {fit(error, 300)}")
    if d.attachments:
        lines.append(f"📎 {fit(d.attachments, 300)}")
    if version.author == "model" and d.note:
        lines.append(f"🤖 <i>{fit(d.note, 300)}</i>")
    if d.kind == "pr":
        lines.append("<i>✅ Post posts this as a comment review: it never approves or merges.</i>")
    if label:
        lines.append(f"🏷 ✅ Post + label also labels the issue <b>{escape(label)}</b>.")
    if "[your decision" in version.text.lower():
        lines.append(
            "<i>⚖️ A decision is left open in this text. Pick an option below, or reply"
            " with your decision: ✅ Post comes once it's settled.</i>"
        )
    lines += [
        f"<pre>{escape(version.text)}</pre>",
        "<i>To change it, reply to this message with what to change. Your own text"
        f" instead: start the reply with {OWN_TEXT}</i>",
    ]
    return "\n".join(lines)


def verdict(d: Draft, v: Verdict) -> str:
    """The assessment before a draft. Every field is capped so it fits one message."""

    repo = d.repo.split("/", 1)[1]
    lines = [
        f"<b>🧭 Assessment #{d.number}</b> · {escape(repo)}",
        fit(d.title, 150),
        f"<b>{CATEGORIES[v.category]}</b> · confidence: {v.confidence}",
    ]
    if v.evidence:
        lines.append("<b>Evidence</b>")
        for e in v.evidence:
            mark = "✓" if e.verified else "⚠️ not found in"
            lines.append(
                f"• {mark} {fit(e.source, 60)}: «{fit(e.quote, 100)}»\n  {fit(e.point, 120)}"
            )
    if v.asks:
        lines.append("<b>Asked</b>")
        for a in v.asks:
            mark = "" if a.verified else "⚠️ not in the message: "
            lines.append(f"• {mark}«{fit(a.quote, 100)}»")
    if v.decision:
        lines.append(f"⚖️ <b>Yours to decide</b>: {fit(v.decision, 200)}")
        lines += [f"  {n}. {fit(o, 80)}" for n, o in enumerate(v.options, 1)]
    if v.missing:
        lines.append("<b>Missing</b>")
        lines += [f"• {fit(m, 110)}" for m in v.missing]
    if v.code:
        lines.append(f"<b>Where</b>: {fit(v.code, 200)}")
    if v.fix:
        lines.append(f"<b>Fix</b>: {fit(v.fix, 350)}")
    if v.unknown_paths:
        lines.append(f"⚠️ No such file in the code: {fit(', '.join(v.unknown_paths), 120)}")
    if v.looked_at:
        lines.append(f"🔎 <i>{len(v.looked_at)} lookups: {fit('; '.join(v.looked_at), 150)}</i>")
    return "\n".join(lines)


def review_verdict(d: Draft, v) -> str:
    """A PR's assessment before its review text (``reviews.ReviewVerdict``); the
    findings are shown while they fit in one message."""

    repo = d.repo.split("/", 1)[1]
    lines = [
        f"<b>🧭 Review of #{d.number}</b> · {escape(repo)}",
        fit(d.title, 150),
        f"<b>{RECOMMENDATIONS[v.recommendation]}</b> · confidence: {v.confidence}",
        f"<i>{fit(v.summary, 400)}</i>",
    ]
    if v.facts:
        lines.append("<b>Checked</b>")
        lines += [f"• {fit(f, 140)}" for f in v.facts[:6]]
    if v.findings:
        lines.append("<b>Findings</b>")
        used = sum(len(x) + 1 for x in lines)
        for shown, f in enumerate(v.findings):
            mark = "" if f.verified else "⚠️ not found: "
            item = (
                f"{SEVERITIES[f.severity]} {mark}{fit(f.where, 70)}: «{fit(f.quote, 80)}»"
                f"\n  {fit(f.point, 160)}"
            )
            if used + len(item) > REVIEW_ROOM:  # Telegram allows 4096 in all
                lines.append(f"…and {len(v.findings) - shown} more (the web UI has them all)")
                break
            lines.append(item)
            used += len(item) + 1
    if v.decision:
        lines.append(f"⚖️ <b>Yours to decide</b>: {fit(v.decision, 200)}")
        lines += [f"  {n}. {fit(o, 80)}" for n, o in enumerate(v.options, 1)]
    if v.missing:
        lines.append("<b>Missing</b>")
        lines += [f"• {fit(m, 110)}" for m in v.missing]
    if v.looked_at:
        lines.append(f"🔎 <i>{len(v.looked_at)} lookups; judged against {fit(v.judged_at, 80)}</i>")
    return "\n".join(lines)


def posted(d: Draft, label: str = "") -> str:
    what = "review of" if d.kind == "pr" else "reply to"
    text = f"✅ <b>Posted</b> the {what} #{d.number} · {escape(d.repo.split('/', 1)[1])}"
    return text + (f" · 🏷 {escape(label)}" if label else "")


# Room for the prompt in a handoff message, escaped (Telegram allows 4096 in all).
HANDOFF_ROOM = 3600


def handoff(d: Draft, prompt: str, confirmed: bool) -> str:
    """A bug's prompt for Claude Code, in a block Telegram copies with one tap."""

    repo = d.repo.split("/", 1)[1]
    status = "confirmed" if confirmed else "suspected, not confirmed"
    lines = [
        f"<b>🐞 For Claude #{d.number}</b> · {escape(repo)} · {status}",
        f"<pre>{fit(prompt, HANDOFF_ROOM)}</pre>",
    ]
    if len(escape(prompt)) > HANDOFF_ROOM:
        lines.append("<i>Shortened here; the web UI has all of it.</i>")
    lines.append("<i>Tap to copy, then paste it into Claude Code in the project.</i>")
    return "\n".join(lines)


def brief(repo: str, label: str, text: str) -> str:
    """A repo brief up for approval. Telegram allows 4096 characters per message."""

    return (
        f"<b>📘 Repo brief</b> · {escape(repo)} · {fit(label, 80)}\n"
        "<i>Written by the local model from the repo's docs. Approve it to use it as"
        " background for summaries and drafts.</i>\n\n" + fit(text, 3600)
    )
