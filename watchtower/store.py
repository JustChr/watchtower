"""The shared SQLite database: what has been seen, where polling stands, the outbox,
the user's button presses and replies (decisions), the model's jobs and the reply
drafts.

Each process -- and each thread -- opens its own ``Store``. WAL mode lets the
watcher write while the gateway reads.

A ``job`` is model work the watcher hands the worker (a summary, a brief, a
replay): ``queued`` → ``running`` → ``done`` / ``failed``.

A draft moves: ``prep`` (worker: it needs one) → ``queued`` (watcher: thread,
files and code fetched) → ``drafting`` → ``ready`` (worker) → ``posting`` →
``posted`` (poster), or → ``rejected`` / ``superseded`` (a newer draft for the
same thread) / ``failed``. Each text the user could approve is a
``draft_version``: the model's first, then each of the user's edits.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS seen (key TEXT PRIMARY KEY, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS cursor (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY,
    topic TEXT NOT NULL,
    text TEXT NOT NULL,
    url TEXT,
    silent INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL,
    sent REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    buttons TEXT,                -- JSON [[label, callback_data], ...]
    ref TEXT,                    -- what the message shows, e.g. "draft:12" (replies to it count)
    message_id INTEGER           -- Telegram's, once sent
);
CREATE TABLE IF NOT EXISTS heartbeat (name TEXT PRIMARY KEY, at REAL NOT NULL, detail TEXT);
CREATE TABLE IF NOT EXISTS decision (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,          -- what was decided on, e.g. "brief", "draft"
    action TEXT NOT NULL,        -- e.g. "approve", "reject", "reply"
    ref INTEGER NOT NULL,
    at REAL NOT NULL,
    applied REAL,
    text TEXT                    -- a reply's text
);
CREATE TABLE IF NOT EXISTS job (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,              -- summary, brief, eval
    key TEXT NOT NULL,               -- what it is about; one open job per key
    payload TEXT NOT NULL,           -- JSON
    status TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS draft (
    id INTEGER PRIMARY KEY,
    event_key TEXT NOT NULL UNIQUE,  -- the activity it answers
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    kind TEXT NOT NULL,              -- where the reply goes: issue, discussion
    topic TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,               -- the activity it answers
    reply_to TEXT NOT NULL,          -- discussion: the top-level comment to reply under
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',   -- the model's note for the user
    reason TEXT NOT NULL DEFAULT '', -- the user's reason for rejecting it
    posted_url TEXT NOT NULL DEFAULT '',
    attachments TEXT NOT NULL DEFAULT '', -- what code found attached, for the user
    verdict TEXT NOT NULL DEFAULT '',     -- the assessment (analysis.Verdict as JSON)
    created REAL NOT NULL,
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS draft_version (
    id INTEGER PRIMARY KEY,
    draft INTEGER NOT NULL,
    text TEXT NOT NULL,
    author TEXT NOT NULL,            -- model, user
    created REAL NOT NULL
);
"""
# Columns added after a table was first deployed: (table, column, type).
MIGRATIONS = (
    ("outbox", "buttons", "TEXT"),
    ("outbox", "ref", "TEXT"),
    ("outbox", "message_id", "INTEGER"),
    ("decision", "text", "TEXT"),
    ("draft", "attachments", "TEXT NOT NULL DEFAULT ''"),
    ("draft", "verdict", "TEXT NOT NULL DEFAULT ''"),
)
# After the migrations: they may index a migrated column.
INDEXES = "CREATE INDEX IF NOT EXISTS outbox_message ON outbox (message_id);"

MAX_ATTEMPTS = 5
_DRAFT_COLUMNS = (
    "id, event_key, repo, number, kind, topic, title, url, reply_to, status, note, reason,"
    " posted_url, attachments, verdict"
)


@dataclass(frozen=True)
class Outgoing:
    id: int
    topic: str
    text: str
    url: str | None
    silent: bool
    attempts: int
    buttons: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Job:
    id: int
    kind: str
    key: str
    payload: dict


@dataclass(frozen=True)
class Draft:
    id: int
    event_key: str
    repo: str
    number: int
    kind: str
    topic: str
    title: str
    url: str
    reply_to: str
    status: str
    note: str
    reason: str
    posted_url: str
    attachments: str
    verdict: str  # JSON; ``analysis.Verdict.from_json``


@dataclass(frozen=True)
class Version:
    id: int
    draft: int
    text: str
    author: str
    number: int  # 1 for the model's, 2 for the first edit, ...


class Store:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        for table, column, kind in MIGRATIONS:
            columns = {row[1] for row in self.db.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                try:
                    self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
                except sqlite3.OperationalError:  # the other process was faster
                    pass
        self.db.executescript(INDEXES)

    def close(self) -> None:
        self.db.close()

    # -- dedup ---------------------------------------------------------------

    def is_seen(self, key: str) -> bool:
        return self.db.execute("SELECT 1 FROM seen WHERE key = ?", (key,)).fetchone() is not None

    def mark_seen(self, key: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (key, time.time()))

    # -- cursors -------------------------------------------------------------

    def get_cursor(self, name: str) -> str | None:
        row = self.db.execute("SELECT value FROM cursor WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_cursor(self, name: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO cursor VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value = excluded.value",
            (name, value),
        )

    # -- outbox --------------------------------------------------------------

    def enqueue(
        self,
        topic: str,
        text: str,
        *,
        url: str | None = None,
        silent: bool = False,
        seen_key: str | None = None,
        buttons: list[tuple[str, str]] | None = None,
        ref: str | None = None,
        done_job: int | None = None,
    ) -> None:
        """Queue a message; with ``seen_key``, mark it seen in the same transaction,
        with ``done_job``, finish that job in it (a restart never reports twice).

        ``buttons`` are ``(label, callback_data)`` pairs, shown in one row. ``ref``
        (``kind:id``) makes the user's Telegram replies to the message count.
        """

        with self.db:
            self.db.execute("BEGIN")
            self.db.execute(
                "INSERT INTO outbox (topic, text, url, silent, created, buttons, ref)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (topic, text, url, int(silent), time.time(), json.dumps(buttons or []), ref),
            )
            if seen_key is not None:
                self.mark_seen(seen_key)
            if done_job is not None:
                self.finish_job(done_job)

    def pending(self, limit: int = 10) -> list[Outgoing]:
        rows = self.db.execute(
            "SELECT id, topic, text, url, silent, attempts, buttons FROM outbox"
            " WHERE sent IS NULL AND attempts < ? ORDER BY id LIMIT ?",
            (MAX_ATTEMPTS, limit),
        ).fetchall()
        return [
            Outgoing(
                r[0],
                r[1],
                r[2],
                r[3],
                bool(r[4]),
                r[5],
                tuple(map(tuple, json.loads(r[6] or "[]"))),
            )
            for r in rows
        ]

    def mark_sent(self, message_id: int, telegram_id: int | None = None) -> None:
        self.db.execute(
            "UPDATE outbox SET sent = ?, message_id = ? WHERE id = ?",
            (time.time(), telegram_id, message_id),
        )

    def ref_for_message(self, telegram_id: int) -> str | None:
        """The ``ref`` of the message Telegram knows as ``telegram_id``, if it has one."""

        row = self.db.execute(
            "SELECT ref FROM outbox WHERE message_id = ? ORDER BY id DESC LIMIT 1", (telegram_id,)
        ).fetchone()
        return row[0] if row else None

    def mark_failed(self, message_id: int, error: str, *, permanent: bool = False) -> None:
        self.db.execute(
            "UPDATE outbox SET attempts = CASE WHEN ? THEN ? ELSE attempts + 1 END, error = ?"
            " WHERE id = ?",
            (permanent, MAX_ATTEMPTS, error[:500], message_id),
        )

    def outbox_stats(self, since: float) -> dict[str, int]:
        pending, failed, sent = self.db.execute(
            "SELECT"
            " SUM(sent IS NULL AND attempts < ?),"
            " SUM(sent IS NULL AND attempts >= ?),"
            " SUM(sent >= ?)"
            " FROM outbox",
            (MAX_ATTEMPTS, MAX_ATTEMPTS, since),
        ).fetchone()
        return {"pending": pending or 0, "failed": failed or 0, "sent": sent or 0}

    # -- decisions (button presses and replies, recorded by the gateway) -----

    def record_decision(self, kind: str, action: str, ref: int, text: str | None = None) -> None:
        self.db.execute(
            "INSERT INTO decision (kind, action, ref, at, text) VALUES (?, ?, ?, ?, ?)",
            (kind, action, ref, time.time(), text),
        )

    def open_decisions(self, kind: str) -> list[tuple[int, str, int, str | None]]:
        """``(id, action, ref, text)`` not yet applied, oldest first."""

        return self.db.execute(
            "SELECT id, action, ref, text FROM decision WHERE kind = ? AND applied IS NULL"
            " ORDER BY id",
            (kind,),
        ).fetchall()

    def mark_applied(self, decision_id: int) -> None:
        self.db.execute("UPDATE decision SET applied = ? WHERE id = ?", (time.time(), decision_id))

    # -- jobs (model work for the worker) ------------------------------------

    def add_job(self, kind: str, key: str, payload: dict, *, seen_key: str | None = None) -> bool:
        """Queue a job unless one for ``key`` is still open; with ``seen_key``, mark
        that seen in the same transaction. Returns whether it was queued."""

        now = time.time()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            open_job = self.db.execute(
                "SELECT 1 FROM job WHERE key = ? AND status IN ('queued', 'running')", (key,)
            ).fetchone()
            if open_job is None:
                self.db.execute(
                    "INSERT INTO job (kind, key, payload, status, created, updated)"
                    " VALUES (?, ?, ?, 'queued', ?, ?)",
                    (kind, key, json.dumps(payload, ensure_ascii=False), now, now),
                )
            if seen_key is not None:
                self.mark_seen(seen_key)
        return open_job is None

    def claim_job(self, kinds: tuple[str, ...]) -> Job | None:
        """The oldest queued job of one of ``kinds``, now ``running``."""

        marks = ", ".join("?" * len(kinds))
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                f"SELECT id, kind, key, payload FROM job WHERE status = 'queued'"
                f" AND kind IN ({marks}) ORDER BY id LIMIT 1",
                kinds,
            ).fetchone()
            if row is None:
                return None
            self.db.execute(
                "UPDATE job SET status = 'running', updated = ? WHERE id = ?",
                (time.time(), row[0]),
            )
        return Job(row[0], row[1], row[2], json.loads(row[3]))

    def finish_job(self, job_id: int, *, failed: bool = False) -> None:
        self.db.execute(
            "UPDATE job SET status = ?, updated = ? WHERE id = ?",
            ("failed" if failed else "done", time.time(), job_id),
        )

    def requeue_jobs(self) -> None:
        """Jobs cut off by a restart go back in the queue."""

        self.db.execute(
            "UPDATE job SET status = 'queued', updated = ? WHERE status = 'running'",
            (time.time(),),
        )

    def jobs(self, status: str) -> list[Job]:
        rows = self.db.execute(
            "SELECT id, kind, key, payload FROM job WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
        return [Job(r[0], r[1], r[2], json.loads(r[3])) for r in rows]

    # -- drafts --------------------------------------------------------------

    def add_draft(
        self,
        event_key: str,
        *,
        repo: str,
        number: int,
        kind: str,
        topic: str,
        title: str,
        url: str,
        reply_to: str = "",
        status: str = "queued",
    ) -> None:
        """Add a draft (``prep``: the watcher fetches its thread first; ``queued``:
        ready for the worker); an event gets at most one."""

        now = time.time()
        self.db.execute(
            "INSERT OR IGNORE INTO draft (event_key, repo, number, kind, topic, title, url,"
            " reply_to, status, created, updated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (event_key, repo, number, kind, topic, title, url, reply_to, status, now, now),
        )

    def prepared(self, draft_id: int) -> None:
        """The watcher fetched what the draft needs: over to the worker."""

        self.db.execute(
            "UPDATE draft SET status = 'queued', updated = ? WHERE id = ? AND status = 'prep'",
            (time.time(), draft_id),
        )

    def draft(self, draft_id: int) -> Draft | None:
        row = self.db.execute(
            f"SELECT {_DRAFT_COLUMNS} FROM draft WHERE id = ?", (draft_id,)
        ).fetchone()
        return Draft(*row) if row else None

    def drafts(self, status: str) -> list[Draft]:
        rows = self.db.execute(
            f"SELECT {_DRAFT_COLUMNS} FROM draft WHERE status = ? ORDER BY id", (status,)
        ).fetchall()
        return [Draft(*row) for row in rows]

    def _set_draft(self, draft_id: int, **fields: str) -> None:
        """Update a draft; column names come from this module's callers only."""

        columns = "".join(f", {name} = ?" for name in fields)
        self.db.execute(
            f"UPDATE draft SET updated = ?{columns} WHERE id = ?",
            (time.time(), *fields.values(), draft_id),
        )

    def claim_draft(self) -> Draft | None:
        """The next thread to draft for, now ``drafting``. When a thread has several
        queued, only the newest is drafted (it sees the whole thread); the rest are
        ``superseded``."""

        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT repo, number FROM draft WHERE status = 'queued' ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            (newest,) = self.db.execute(
                "SELECT MAX(id) FROM draft WHERE status = 'queued' AND repo = ? AND number = ?",
                row,
            ).fetchone()
            self.db.execute(
                "UPDATE draft SET status = 'superseded', updated = ?"
                " WHERE status = 'queued' AND repo = ? AND number = ? AND id < ?",
                (time.time(), *row, newest),
            )
            self._set_draft(newest, status="drafting")
        return self.draft(newest)

    def requeue_drafting(self) -> None:
        """Drafts cut off by a restart go back in the queue."""

        self.db.execute(
            "UPDATE draft SET status = 'queued', updated = ? WHERE status = 'drafting'",
            (time.time(),),
        )

    def finish_draft(
        self, draft_id: int, text: str, note: str, attachments: str = "", verdict: str = ""
    ) -> Version:
        """The model's draft is ready; an older ready draft for the same thread is outdated."""

        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            draft = self.draft(draft_id)
            self.db.execute(
                "UPDATE draft SET status = 'superseded', updated = ?"
                " WHERE status = 'ready' AND repo = ? AND number = ? AND id != ?",
                (time.time(), draft.repo, draft.number, draft_id),
            )
            self._set_draft(
                draft_id, status="ready", note=note, attachments=attachments, verdict=verdict
            )
            version = self.add_version(draft_id, text, "model")
        return version

    def fail_draft(self, draft_id: int) -> None:
        self._set_draft(draft_id, status="failed")

    def add_version(self, draft_id: int, text: str, author: str) -> Version:
        version_id = self.db.execute(
            "INSERT INTO draft_version (draft, text, author, created) VALUES (?, ?, ?, ?)",
            (draft_id, text, author, time.time()),
        ).lastrowid
        return self.version(version_id)

    def version(self, version_id: int) -> Version | None:
        row = self.db.execute(
            "SELECT v.id, v.draft, v.text, v.author,"
            " (SELECT COUNT(*) FROM draft_version w WHERE w.draft = v.draft AND w.id <= v.id)"
            " FROM draft_version v WHERE v.id = ?",
            (version_id,),
        ).fetchone()
        return Version(*row) if row else None

    def latest_version(self, draft_id: int) -> Version | None:
        row = self.db.execute(
            "SELECT MAX(id) FROM draft_version WHERE draft = ?", (draft_id,)
        ).fetchone()
        return self.version(row[0]) if row[0] is not None else None

    def begin_post(self, draft_id: int, decision_id: int) -> bool:
        """Claim a ready draft for posting and consume the decision, in one step: after
        a crash the draft is left ``posting``, never posted twice."""

        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            claimed = self.db.execute(
                "UPDATE draft SET status = 'posting', updated = ? WHERE id = ? AND status = 'ready'",
                (time.time(), draft_id),
            ).rowcount
            self.mark_applied(decision_id)
        return bool(claimed)

    def posted(self, draft_id: int, url: str) -> None:
        self._set_draft(draft_id, status="posted", posted_url=url)

    def post_failed(self, draft_id: int) -> None:
        """Posting failed for sure (GitHub said no): the draft can be approved again."""

        self._set_draft(draft_id, status="ready")

    def reject_draft(self, draft_id: int) -> None:
        self._set_draft(draft_id, status="rejected")

    def set_reason(self, draft_id: int, reason: str) -> None:
        self._set_draft(draft_id, reason=reason)

    def interrupted_posts(self) -> list[Draft]:
        """Drafts a restart cut off while posting, now ``failed``: whether GitHub got
        them is unknown, so they are never retried automatically."""

        drafts = self.drafts("posting")
        for draft in drafts:
            self._set_draft(draft.id, status="failed")
        return drafts

    # -- heartbeats ----------------------------------------------------------

    def beat(self, name: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO heartbeat VALUES (?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET at = excluded.at, detail = excluded.detail",
            (name, time.time(), detail),
        )

    def forget_beat(self, name: str) -> None:
        """A process that no longer exists leaves /status."""

        self.db.execute("DELETE FROM heartbeat WHERE name = ?", (name,))

    def heartbeats(self) -> dict[str, tuple[float, str]]:
        rows = self.db.execute("SELECT name, at, detail FROM heartbeat").fetchall()
        return {name: (at, detail or "") for name, at, detail in rows}
