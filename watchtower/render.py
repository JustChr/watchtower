"""Telegram message text (HTML parse mode). Every value from GitHub or the model is escaped."""

from __future__ import annotations

import re
from html import escape

from .events import Event
from .llm import Summary
from .store import Draft, Version

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
SNIPPET = 280

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


def draft(d: Draft, version: Version, error: str = "") -> str:
    """A draft up for approval. The text is shown in full -- it is exactly what gets
    posted -- so callers keep it within ``drafts.MAX_SHOWN``."""

    repo = d.repo.split("/", 1)[1]
    head = f"<b>✍️ Draft reply #{d.number}</b> · {escape(repo)}"
    if version.author == "user":
        head += f" · v{version.number}, your edit"
    lines = [head, fit(d.title, 200)]
    if error:
        lines.append(f"⚠️ {fit(error, 300)}")
    if d.attachments:
        lines.append(f"📎 {fit(d.attachments, 300)}")
    if version.author == "model" and d.note:
        lines.append(f"🤖 <i>{fit(d.note, 300)}</i>")
    lines += [
        f"<pre>{escape(version.text)}</pre>",
        "<i>To change it, reply to this message with your version.</i>",
    ]
    return "\n".join(lines)


def posted(d: Draft) -> str:
    return f"✅ <b>Posted</b> the reply to #{d.number} · {escape(d.repo.split('/', 1)[1])}"


def brief(repo: str, label: str, text: str) -> str:
    """A repo brief up for approval. Telegram allows 4096 characters per message."""

    return (
        f"<b>📘 Repo brief</b> · {escape(repo)} · {fit(label, 80)}\n"
        "<i>Written by the local model from the repo's docs. Approve it to use it as"
        " background for summaries and drafts.</i>\n\n" + fit(text, 3600)
    )
