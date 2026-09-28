"""Telegram message text (HTML parse mode). Every value from GitHub or the model is escaped."""

from __future__ import annotations

import re
from html import escape

from .events import Event
from .llm import Summary

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


def message(event: Event, summary: Summary | None) -> str:
    repo = event.repo.split("/", 1)[1]
    lines = [
        f"<b>{HEADINGS[event.kind]} #{event.number}</b> · {escape(repo)}",
        escape(event.title),
        f"by {escape(event.author)}",
    ]
    if summary is not None:
        label = KIND_LABELS[summary.kind]
        if summary.needs_reply:
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


def brief(repo: str, label: str, text: str) -> str:
    """A repo brief up for approval. Telegram allows 4096 characters per message."""

    return (
        f"<b>📘 Repo brief</b> · {escape(repo)} · {fit(label, 80)}\n"
        "<i>Written by the local model from the repo's docs. Approve it to use it as"
        " background for summaries and drafts.</i>\n\n" + fit(text, 3600)
    )
