"""The shared SQLite database: what has been seen, where polling stands, the outbox.

Each process -- and each thread -- opens its own ``Store``. WAL mode lets the
watcher write while the gateway reads.
"""

from __future__ import annotations

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
    error TEXT
);
CREATE TABLE IF NOT EXISTS heartbeat (name TEXT PRIMARY KEY, at REAL NOT NULL, detail TEXT);
"""

MAX_ATTEMPTS = 5


@dataclass(frozen=True)
class Outgoing:
    id: int
    topic: str
    text: str
    url: str | None
    silent: bool
    attempts: int


class Store:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

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
    ) -> None:
        """Queue a message; with ``seen_key``, mark it seen in the same transaction."""

        with self.db:
            self.db.execute("BEGIN")
            self.db.execute(
                "INSERT INTO outbox (topic, text, url, silent, created) VALUES (?, ?, ?, ?, ?)",
                (topic, text, url, int(silent), time.time()),
            )
            if seen_key is not None:
                self.mark_seen(seen_key)

    def pending(self, limit: int = 10) -> list[Outgoing]:
        rows = self.db.execute(
            "SELECT id, topic, text, url, silent, attempts FROM outbox"
            " WHERE sent IS NULL AND attempts < ? ORDER BY id LIMIT ?",
            (MAX_ATTEMPTS, limit),
        ).fetchall()
        return [Outgoing(r[0], r[1], r[2], r[3], bool(r[4]), r[5]) for r in rows]

    def mark_sent(self, message_id: int) -> None:
        self.db.execute("UPDATE outbox SET sent = ? WHERE id = ?", (time.time(), message_id))

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

    # -- heartbeats ----------------------------------------------------------

    def beat(self, name: str, detail: str = "") -> None:
        self.db.execute(
            "INSERT INTO heartbeat VALUES (?, ?, ?)"
            " ON CONFLICT(name) DO UPDATE SET at = excluded.at, detail = excluded.detail",
            (name, time.time(), detail),
        )

    def heartbeats(self) -> dict[str, tuple[float, str]]:
        rows = self.db.execute("SELECT name, at, detail FROM heartbeat").fetchall()
        return {name: (at, detail or "") for name, at, detail in rows}
