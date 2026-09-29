"""A trace of every model call: what was asked, what came back, how long it took.

The worker records each call to Ollama here (``llm.tracer``), with what it was
for (``subject``: ``draft:12``, ``event:<key>``, ``brief:<repo>:<ref>``,
``eval:...``) and the step it was in ("investigating, step 3"). The web UI shows
it, so every draft can be followed call by call: the full prompt the model saw
and the full answer it gave. Calls older than ``KEEP_DAYS`` are pruned.

Prompts hold strangers' text and model output: the trace stores them as data,
and whoever shows them escapes them.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

KEEP_DAYS = 90

SCHEMA = """
CREATE TABLE IF NOT EXISTS call (
    id INTEGER PRIMARY KEY,
    at REAL NOT NULL,                -- when it started
    seconds REAL NOT NULL,
    subject TEXT NOT NULL,           -- what it was for: draft:12, event:<key>, ...
    step TEXT NOT NULL,              -- e.g. "investigating, step 3"
    kind TEXT NOT NULL,              -- summary, findings, investigate, assess, reply, brief
    model TEXT NOT NULL,
    num_ctx INTEGER NOT NULL,
    prompt_chars INTEGER NOT NULL,
    answer_chars INTEGER NOT NULL,
    tool_calls INTEGER NOT NULL,
    request TEXT NOT NULL,           -- JSON: messages, tool names, format
    answer TEXT NOT NULL,            -- JSON: the model's message
    error TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS call_subject ON call (subject);
CREATE INDEX IF NOT EXISTS call_at ON call (at);
"""

_SUMMARY_COLUMNS = (
    "id, at, seconds, subject, step, kind, model, num_ctx, prompt_chars, answer_chars,"
    " tool_calls, error"
)


def kind_of(payload: dict[str, Any]) -> str:
    """Which pass a request is, from its shape: the tools, or the answer's schema."""

    if payload.get("tools"):
        return "investigate"
    fields = set(((payload.get("format") or {}).get("properties") or {}))
    if "needs_reply" in fields:
        return "summary"
    if "category" in fields:
        return "assess"
    if "findings" in fields:
        return "findings"
    if "reply" in fields:
        return "reply"
    return "brief"


def _chars(message: dict) -> int:
    return len(message.get("content") or "") + len(message.get("thinking") or "")


class Trace:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.subject = ""
        self.step = ""

    def close(self) -> None:
        self.db.close()

    def record(
        self,
        payload: dict[str, Any],
        answer: dict | None,
        started: float,
        seconds: float,
        error: str = "",
    ) -> None:
        """One call, as ``llm`` made it. Never raises: a trace mustn't break a draft."""

        try:
            messages = payload.get("messages") or []
            message = (answer or {}).get("message") or {}
            request = {
                "messages": messages,
                "tools": [t["function"]["name"] for t in payload.get("tools") or []],
                "format": payload.get("format"),
                "options": payload.get("options"),
            }
            self.db.execute(
                "INSERT INTO call (at, seconds, subject, step, kind, model, num_ctx,"
                " prompt_chars, answer_chars, tool_calls, request, answer, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    started,
                    seconds,
                    self.subject,
                    self.step,
                    kind_of(payload),
                    str(payload.get("model", "")),
                    int((payload.get("options") or {}).get("num_ctx", 0)),
                    sum(_chars(m) for m in messages),
                    _chars(message),
                    len(message.get("tool_calls") or []),
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(message, ensure_ascii=False),
                    error[:500],
                ),
            )
        except Exception:  # noqa: BLE001 -- losing a trace row is better than losing a draft
            pass

    def prune(self, now: float | None = None) -> int:
        cutoff = (time.time() if now is None else now) - KEEP_DAYS * 86400
        return self.db.execute("DELETE FROM call WHERE at < ?", (cutoff,)).rowcount

    # -- reading (the web UI) ------------------------------------------------

    def calls(
        self, subject: str | None = None, before: int | None = None, limit: int = 100
    ) -> list[dict]:
        """Newest first, without the full request and answer."""

        where, args = [], []
        if subject is not None:
            where.append("subject = ?")
            args.append(subject)
        if before is not None:
            where.append("id < ?")
            args.append(before)
        sql = f"SELECT {_SUMMARY_COLUMNS} FROM call"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self.db.execute(f"{sql} ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        names = [c.strip() for c in _SUMMARY_COLUMNS.split(",")]
        return [dict(zip(names, row, strict=True)) for row in rows]

    def call(self, call_id: int) -> dict | None:
        row = self.db.execute(
            f"SELECT {_SUMMARY_COLUMNS}, request, answer FROM call WHERE id = ?", (call_id,)
        ).fetchone()
        if row is None:
            return None
        names = [c.strip() for c in _SUMMARY_COLUMNS.split(",")] + ["request", "answer"]
        found = dict(zip(names, row, strict=True))
        found["request"] = json.loads(found["request"])
        found["answer"] = json.loads(found["answer"])
        return found

    def daily(self, since: float) -> list[dict]:
        """Per day and kind: calls, seconds, prompt characters."""

        rows = self.db.execute(
            "SELECT date(at, 'unixepoch') AS day, kind, COUNT(*), SUM(seconds), SUM(prompt_chars)"
            " FROM call WHERE at >= ? GROUP BY day, kind ORDER BY day",
            (since,),
        ).fetchall()
        return [
            {"day": d, "kind": k, "calls": n, "seconds": s, "prompt_chars": p}
            for d, k, n, s, p in rows
        ]
