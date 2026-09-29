"""A searchable local copy of each watched repo's issues, PRs, discussions and comments.

This is what later agents learn a repo from: earlier reports of the same
problem, how they were resolved, and how the maintainers answered. The watcher
keeps it in sync (the first sync pulls the whole history); the worker reads it
from the shared volume, so it never needs a token or the internet.

Everything here was written on GitHub and stays untrusted data. The one field
GitHub computes itself is ``association`` (OWNER, MEMBER, CONTRIBUTOR, ...):
it is how maintainer-written text is told apart from strangers' text.

The same file holds the repo briefs (see ``brief``) with their approval state,
and the release list (with the maintainers' notes), for the offline worker. The worker writes the briefs it's asked for.

Issues, PRs and discussions share one number space per repo, so an item is
``(repo, number)``. Syncs upsert, so overlapping pages are harmless.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .events import Source

EPOCH = "1970-01-01T00:00:00Z"
MAX_TEXT = 20_000
# Upper bound on requests per list and sync; the next sync carries on.
ROUNDS = 20
MAINTAINERS = frozenset({"OWNER", "MEMBER", "COLLABORATOR"})

SCHEMA = """
CREATE TABLE IF NOT EXISTS item (
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    kind TEXT NOT NULL,          -- issue, pr, discussion
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    association TEXT NOT NULL,
    state TEXT NOT NULL,         -- open, completed, not_planned, duplicate, merged, closed, answered
    labels TEXT NOT NULL,        -- comma-separated; a discussion's category
    body TEXT NOT NULL,
    url TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    PRIMARY KEY (repo, number)
);
CREATE TABLE IF NOT EXISTS comment (
    id TEXT PRIMARY KEY,         -- "issue-<id>" or "discussion-<node id>"
    repo TEXT NOT NULL,
    number INTEGER NOT NULL,
    author TEXT NOT NULL,
    association TEXT NOT NULL,
    body TEXT NOT NULL,
    url TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS comment_thread ON comment (repo, number, created);
CREATE TABLE IF NOT EXISTS cursor (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS brief (
    id INTEGER PRIMARY KEY,
    repo TEXT NOT NULL,
    ref TEXT NOT NULL,           -- the release tag it describes, or a main commit id
    label TEXT NOT NULL,         -- e.g. "v2.4.0b2 (beta)", "main @ 1a2b3c4"
    text TEXT NOT NULL,
    status TEXT NOT NULL,        -- pending, approved, rejected, failed
    created REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS release (
    repo TEXT NOT NULL,
    tag TEXT NOT NULL,
    prerelease INTEGER NOT NULL,
    published TEXT NOT NULL,
    notes TEXT NOT NULL,         -- the maintainers' release notes
    PRIMARY KEY (repo, tag)
);
CREATE VIRTUAL TABLE IF NOT EXISTS item_fts USING fts5(
    title, body, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS comment_fts USING fts5(
    body, content='', contentless_delete=1, tokenize='unicode61 remove_diacritics 2'
);
"""

_DISCUSSION_FIELDS = """
        number title url body createdAt updatedAt closed isAnswered
        category { name } author { login } authorAssociation
        comments(first: 100) {
          nodes {
            id url body createdAt updatedAt author { login } authorAssociation
            replies(first: 50) {
              nodes { id url body createdAt updatedAt author { login } authorAssociation }
            }
          }
        }
"""
DISCUSSIONS_QUERY = (
    """
query($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 25, after: $after, orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { %s }
    }
  }
}
"""
    % _DISCUSSION_FIELDS
)
DISCUSSION_QUERY = (
    """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    discussion(number: $number) { %s }
  }
}
"""
    % _DISCUSSION_FIELDS
)

_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Release:
    tag: str
    prerelease: bool
    published: str
    notes: str

    @property
    def label(self) -> str:
        return f"{self.tag} ({'beta' if self.prerelease else 'stable'})"


@dataclass(frozen=True)
class Brief:
    id: int
    repo: str
    ref: str
    label: str
    text: str
    status: str
    created: float


@dataclass(frozen=True)
class Hit:
    number: int
    kind: str
    title: str
    state: str
    url: str


class History:
    def __init__(self, path: Path | str) -> None:
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def get_cursor(self, name: str) -> str | None:
        row = self.db.execute("SELECT value FROM cursor WHERE name = ?", (name,)).fetchone()
        return row[0] if row else None

    def set_cursor(self, name: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO cursor VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value = excluded.value",
            (name, value),
        )

    # -- writing -------------------------------------------------------------

    def _upsert(self, table: str, key: str, row: dict[str, object]) -> int:
        """Insert or update ``row`` (column names come from this module only); returns its rowid."""

        columns = ", ".join(row)
        marks = ", ".join("?" * len(row))
        updates = ", ".join(f"{c} = excluded.{c}" for c in row)
        (rowid,) = self.db.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({marks})"
            f" ON CONFLICT({key}) DO UPDATE SET {updates} RETURNING rowid",
            tuple(row.values()),
        ).fetchone()
        return rowid

    def put_item(self, repo: str, number: int, **fields: str) -> None:
        fields["body"] = fields["body"][:MAX_TEXT]
        rowid = self._upsert("item", "repo, number", {"repo": repo, "number": number, **fields})
        self.db.execute("DELETE FROM item_fts WHERE rowid = ?", (rowid,))
        self.db.execute(
            "INSERT INTO item_fts (rowid, title, body) VALUES (?, ?, ?)",
            (rowid, fields["title"], fields["body"]),
        )

    def put_comment(self, cid: str, repo: str, number: int, **fields: str) -> None:
        fields["body"] = fields["body"][:MAX_TEXT]
        rowid = self._upsert("comment", "id", {"id": cid, "repo": repo, "number": number, **fields})
        self.db.execute("DELETE FROM comment_fts WHERE rowid = ?", (rowid,))
        self.db.execute(
            "INSERT INTO comment_fts (rowid, body) VALUES (?, ?)", (rowid, fields["body"])
        )

    # -- briefs --------------------------------------------------------------

    def add_brief(self, repo: str, ref: str, label: str, text: str | None) -> int:
        """Record a new brief (``None``: generation failed); returns its id."""

        return self.db.execute(
            "INSERT INTO brief (repo, ref, label, text, status, created) VALUES (?, ?, ?, ?, ?, ?)",
            (repo, ref, label, text or "", "pending" if text else "failed", time.time()),
        ).lastrowid

    def latest_brief(self, repo: str) -> Brief | None:
        row = self.db.execute(
            "SELECT id, repo, ref, label, text, status, created FROM brief"
            " WHERE repo = ? ORDER BY id DESC LIMIT 1",
            (repo,),
        ).fetchone()
        return Brief(*row) if row else None

    def decide_brief(self, brief_id: int, approved: bool) -> None:
        """Only a pending brief can be decided; failed ones stay failed."""

        self.db.execute(
            "UPDATE brief SET status = ? WHERE id = ? AND status = 'pending'",
            ("approved" if approved else "rejected", brief_id),
        )

    def approved_brief(self, repo: str) -> str | None:
        row = self.db.execute(
            "SELECT text FROM brief WHERE repo = ? AND status = 'approved' ORDER BY id DESC LIMIT 1",
            (repo,),
        ).fetchone()
        return row[0] if row else None

    # -- releases (for drafts: which version is current, what changed since) ----

    def put_releases(self, repo: str, releases: Sequence[Release]) -> None:
        with self.db:
            self.db.execute("BEGIN")
            self.db.execute("DELETE FROM release WHERE repo = ?", (repo,))
            self.db.executemany(
                "INSERT OR REPLACE INTO release VALUES (?, ?, ?, ?, ?)",
                [(repo, r.tag, int(r.prerelease), r.published, r.notes) for r in releases],
            )

    def releases(self, repo: str) -> list[Release]:
        """Newest first."""

        rows = self.db.execute(
            "SELECT tag, prerelease, published, notes FROM release WHERE repo = ?"
            " ORDER BY published DESC",
            (repo,),
        ).fetchall()
        return [Release(tag, bool(pre), published, notes) for tag, pre, published, notes in rows]

    # -- reading -------------------------------------------------------------

    def closed(self, repo: str, kind: str = "issue") -> list[int]:
        """Numbers of the closed items of ``kind``, newest first."""

        rows = self.db.execute(
            "SELECT number FROM item WHERE repo = ? AND kind = ? AND state != 'open'"
            " ORDER BY number DESC",
            (repo, kind),
        ).fetchall()
        return [number for (number,) in rows]

    def counts(self, repo: str) -> dict[str, int]:
        rows = self.db.execute(
            "SELECT kind, COUNT(*) FROM item WHERE repo = ? GROUP BY kind", (repo,)
        ).fetchall()
        counts = {"issue": 0, "pr": 0, "discussion": 0, **dict(rows)}
        (counts["comment"],) = self.db.execute(
            "SELECT COUNT(*) FROM comment WHERE repo = ?", (repo,)
        ).fetchone()
        return counts

    def search(self, repo: str, query: str, limit: int = 10) -> list[Hit]:
        """Items whose title, body or comments match any word of ``query``, best first.

        ``query`` may come from a model, so it is reduced to plain words and never
        reaches FTS5's query syntax.
        """

        words = _WORD.findall(query)[:32]
        if not words:
            return []
        match = " OR ".join(f'"{w}"' for w in words)
        rows = self.db.execute(
            "SELECT item.number, bm25(item_fts, 3.0, 1.0) FROM item_fts"
            " JOIN item ON item.rowid = item_fts.rowid"
            " WHERE item_fts MATCH ? AND item.repo = ?"
            " UNION ALL "
            "SELECT comment.number, bm25(comment_fts) FROM comment_fts"
            " JOIN comment ON comment.rowid = comment_fts.rowid"
            " WHERE comment_fts MATCH ? AND comment.repo = ?",
            (match, repo, match, repo),
        ).fetchall()
        best: dict[int, float] = {}
        for number, score in rows:
            best[number] = min(score, best.get(number, score))
        hits = []
        for number in sorted(best, key=best.__getitem__):
            row = self.db.execute(
                "SELECT kind, title, state, url FROM item WHERE repo = ? AND number = ?",
                (repo, number),
            ).fetchone()
            if row:  # a comment can arrive before its item
                hits.append(Hit(number, *row))
            if len(hits) == limit:
                break
        return hits

    def thread(self, repo: str, number: int) -> dict | None:
        """One item with its comments, oldest first; ``maintainer`` flags maintainer text."""

        self.db.row_factory = sqlite3.Row
        try:
            item = self.db.execute(
                "SELECT * FROM item WHERE repo = ? AND number = ?", (repo, number)
            ).fetchone()
            if item is None:
                return None
            comments = self.db.execute(
                "SELECT author, association, body, url, created FROM comment"
                " WHERE repo = ? AND number = ? ORDER BY created, id",
                (repo, number),
            ).fetchall()
        finally:
            self.db.row_factory = None
        return {
            **dict(item),
            "maintainer": item["association"] in MAINTAINERS,
            "comments": [
                {**dict(c), "maintainer": c["association"] in MAINTAINERS} for c in comments
            ],
        }


# -- sync ----------------------------------------------------------------------


def _login(node: dict | None) -> str:
    return (node or {}).get("login") or "ghost"


def _issue_state(item: dict) -> str:
    if item["state"] == "open":
        return "open"
    if (item.get("pull_request") or {}).get("merged_at"):
        return "merged"
    return item.get("state_reason") or "closed"


def _save_issue(history: History, repo: str, item: dict) -> None:
    history.put_item(
        repo,
        item["number"],
        kind="pr" if "pull_request" in item else "issue",
        title=item["title"],
        author=_login(item.get("user")),
        association=item.get("author_association") or "NONE",
        state=_issue_state(item),
        labels=",".join(label["name"] for label in item.get("labels") or []),
        body=item.get("body") or "",
        url=item["html_url"],
        created=item["created_at"],
        updated=item["updated_at"],
    )


def _save_issue_comment(history: History, repo: str, comment: dict) -> None:
    history.put_comment(
        f"issue-{comment['id']}",
        repo,
        int(comment["issue_url"].rsplit("/", 1)[1]),
        author=_login(comment.get("user")),
        association=comment.get("author_association") or "NONE",
        body=comment.get("body") or "",
        url=comment["html_url"],
        created=comment["created_at"],
        updated=comment["updated_at"],
    )


def _save_discussion(history: History, repo: str, node: dict) -> int:
    number = node["number"]
    state = "answered" if node.get("isAnswered") else "closed" if node.get("closed") else "open"
    history.put_item(
        repo,
        number,
        kind="discussion",
        title=node["title"],
        author=_login(node.get("author")),
        association=node.get("authorAssociation") or "NONE",
        state=state,
        labels=(node.get("category") or {}).get("name") or "",
        body=node.get("body") or "",
        url=node["url"],
        created=node["createdAt"],
        updated=node["updatedAt"],
    )
    saved = 0
    for comment in (node.get("comments") or {}).get("nodes") or []:
        replies = (comment.get("replies") or {}).get("nodes") or []
        for reply in (comment, *replies):
            history.put_comment(
                f"discussion-{reply['id']}",
                repo,
                number,
                author=_login(reply.get("author")),
                association=reply.get("authorAssociation") or "NONE",
                body=reply.get("body") or "",
                url=reply["url"],
                created=reply["createdAt"],
                updated=reply["updatedAt"],
            )
            saved += 1
    return saved


def _sync_list(source: Source, history: History, repo: str, what: str, save) -> int:
    """Pull one REST list oldest-first from its cursor until nothing newer comes back.

    The last request repeats the cursor's URL, so the next sync gets its ETag: a
    free 304 while nothing changed.
    """

    name = f"{what}:{repo}"
    since = history.get_cursor(name) or EPOCH
    path = f"/repos/{repo}/issues" if what == "issues" else f"/repos/{repo}/issues/comments"
    params = {"state": "all"} if what == "issues" else {}
    saved = 0
    for _ in range(ROUNDS):
        # No ``since`` before the first cursor: the issues list answers ``since=1970-…``
        # with nothing at all (the comments list doesn't), so the backfill got no issues.
        after = {"since": since} if since != EPOCH else {}
        items = source.get_list(
            path, sort="updated", direction="asc", per_page=100, **after, **params
        )
        newest = since
        with history.db:
            history.db.execute("BEGIN")
            for item in items:
                save(history, repo, item)
                newest = max(newest, item["updated_at"])
            if newest != since:
                history.set_cursor(name, newest)
        saved += sum(item["updated_at"] > since for item in items)  # not the overlap
        if newest == since:
            break
        since = newest
    return saved


def _sync_discussions(source: Source, history: History, repo: str) -> int:
    """Newest-updated first, down to the cursor. The first page is always refreshed.

    The cursor only moves once a sync reaches it, so a backfill cut short by
    ``ROUNDS`` never skips older discussions.
    """

    name = f"discussions:{repo}"
    since = history.get_cursor(name) or EPOCH
    owner, repo_name = repo.split("/")
    after = None
    newest = since
    saved = 0
    for _ in range(ROUNDS):
        variables = {"owner": owner, "name": repo_name, "after": after}
        data = source.graphql(DISCUSSIONS_QUERY, variables)
        page = ((data.get("repository") or {}).get("discussions")) or {}
        nodes = page.get("nodes") or []
        with history.db:
            history.db.execute("BEGIN")
            for node in nodes:
                saved += 1 + _save_discussion(history, repo, node)
                newest = max(newest, node["updatedAt"])
        info = page.get("pageInfo") or {}
        if not info.get("hasNextPage") or (nodes and nodes[-1]["updatedAt"] <= since):
            if newest != since:
                history.set_cursor(name, newest)
            break
        after = info.get("endCursor")
    return saved


def sync_thread(source: Source, history: History, repo: str, number: int, kind: str) -> None:
    """Refresh one issue or discussion with all its comments right now, so a draft
    sees the thread as it is and not as of the last full sync."""

    if kind == "discussion":
        owner, repo_name = repo.split("/")
        variables = {"owner": owner, "name": repo_name, "number": number}
        node = (source.graphql(DISCUSSION_QUERY, variables).get("repository") or {}).get(
            "discussion"
        )
        if node:
            with history.db:
                history.db.execute("BEGIN")
                _save_discussion(history, repo, node)
        return
    item = source.get_json(f"/repos/{repo}/issues/{number}")
    comments = source.get_list(f"/repos/{repo}/issues/{number}/comments", per_page=100)
    with history.db:
        history.db.execute("BEGIN")
        _save_issue(history, repo, item)
        for comment in comments:
            _save_issue_comment(history, repo, comment)


def sync_repo(source: Source, history: History, repo: str) -> int:
    """Bring one repo's history up to date. Returns how many records were written."""

    written = _sync_list(source, history, repo, "issues", _save_issue)
    written += _sync_list(source, history, repo, "comments", _save_issue_comment)
    written += _sync_discussions(source, history, repo)
    return written
