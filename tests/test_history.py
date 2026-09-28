"""The searchable history: full backfill, incremental sync, search, threads."""

from __future__ import annotations

import pytest

from watchtower import __main__ as cli
from watchtower import config, history, watcher
from watchtower.history import History, sync_repo

REPO = "owner/repo"


class FakeGitHub:
    def __init__(self) -> None:
        self.issues: list[dict] = []
        self.comments: list[dict] = []
        self.discussions: list[dict] = []
        self.page_size = 25
        self.calls: list[str] = []

    def get_list(self, path: str, **params: object) -> list[dict]:
        self.calls.append(path)
        items = self.comments if path.endswith("/comments") else self.issues
        since = params.get("since", "")
        if path.endswith("/issues") and str(since).startswith("1970"):
            return []  # what GitHub really does (seen 2026-09-28)
        return sorted(
            (i for i in items if i["updated_at"] >= since),
            key=lambda i: i["updated_at"],
        )

    def graphql(self, query: str, variables: dict) -> dict:
        self.calls.append("graphql")
        nodes = sorted(self.discussions, key=lambda d: d["updatedAt"], reverse=True)
        start = int(variables["after"] or 0)
        page = nodes[start : start + self.page_size]
        more = start + self.page_size < len(nodes)
        return {
            "repository": {
                "discussions": {
                    "nodes": page,
                    "pageInfo": {"hasNextPage": more, "endCursor": str(start + self.page_size)},
                }
            }
        }


def issue(number, updated, *, title="Title", body="Body", pr=None, **extra) -> dict:
    item = {
        "number": number,
        "title": title,
        "user": {"login": "stranger"},
        "author_association": "NONE",
        "state": "open",
        "labels": [{"name": "bug"}],
        "body": body,
        "created_at": updated,
        "updated_at": updated,
        "html_url": f"https://github.com/{REPO}/issues/{number}",
        **extra,
    }
    if pr is not None:
        item["pull_request"] = {"merged_at": pr}
    return item


def comment(cid, number, updated, body, *, association="NONE") -> dict:
    return {
        "id": cid,
        "issue_url": f"https://api.github.com/repos/{REPO}/issues/{number}",
        "html_url": f"https://github.com/{REPO}/issues/{number}#issuecomment-{cid}",
        "user": {"login": "JustChr" if association == "OWNER" else "stranger"},
        "author_association": association,
        "body": body,
        "created_at": updated,
        "updated_at": updated,
    }


def discussion(number, updated, *, body="", comments=(), answered=False) -> dict:
    return {
        "number": number,
        "title": f"Discussion {number}",
        "url": f"https://github.com/{REPO}/discussions/{number}",
        "body": body,
        "createdAt": updated,
        "updatedAt": updated,
        "closed": False,
        "isAnswered": answered,
        "category": {"name": "Q&A"},
        "author": {"login": "asker"},
        "authorAssociation": "NONE",
        "comments": {"nodes": list(comments)},
    }


@pytest.fixture
def hist(tmp_path):
    h = History(tmp_path / "history.db")
    yield h
    h.close()


def test_first_sync_pulls_the_whole_history(hist):
    gh = FakeGitHub()
    gh.issues = [
        issue(1, "2024-01-01T00:00:00Z", state="closed", state_reason="not_planned"),
        issue(2, "2024-02-01T00:00:00Z", pr="2024-02-02T00:00:00Z", state="closed"),
    ]
    gh.comments = [comment(10, 1, "2024-01-02T00:00:00Z", "Thanks")]
    reply = {
        "id": "R1",
        "url": "u",
        "body": "Use the other endpoint",
        "createdAt": "2024-03-02T00:00:00Z",
        "updatedAt": "2024-03-02T00:00:00Z",
        "author": {"login": "JustChr"},
        "authorAssociation": "OWNER",
    }
    gh.discussions = [
        discussion(3, "2024-03-01T00:00:00Z", comments=[{**reply, "id": "C1", "replies": {}}]),
    ]
    gh.discussions[0]["comments"]["nodes"][0]["replies"] = {"nodes": [reply]}

    assert sync_repo(gh, hist, REPO) == 6
    assert hist.counts(REPO) == {"issue": 1, "pr": 1, "discussion": 1, "comment": 3}
    assert [hist.thread(REPO, n)["state"] for n in (1, 2, 3)] == ["not_planned", "merged", "open"]


def test_resync_updates_in_place_and_follows_the_cursor(hist):
    gh = FakeGitHub()
    gh.issues = [issue(1, "2024-01-01T00:00:00Z", title="Old title")]
    sync_repo(gh, hist, REPO)
    gh.issues = [
        issue(1, "2024-05-01T00:00:00Z", title="New title"),
        issue(2, "2024-05-02T00:00:00Z"),
    ]
    sync_repo(gh, hist, REPO)
    assert hist.counts(REPO)["issue"] == 2
    assert hist.thread(REPO, 1)["title"] == "New title"
    assert hist.get_cursor(f"issues:{REPO}") == "2024-05-02T00:00:00Z"
    assert [h.number for h in hist.search(REPO, "old")] == []


def test_list_sync_ends_on_the_cursor_url_and_stops_when_stuck(hist, monkeypatch):
    monkeypatch.setattr(history, "ROUNDS", 50)
    gh = FakeGitHub()
    # Same timestamp everywhere: the cursor can't move past it, so the sync must stop.
    gh.issues = [issue(n, "2024-01-01T00:00:00Z") for n in range(1, 4)]
    sync_repo(gh, hist, REPO)
    assert gh.calls.count(f"/repos/{REPO}/issues") == 2


def test_discussion_backfill_cut_short_keeps_the_cursor(hist, monkeypatch):
    monkeypatch.setattr(history, "ROUNDS", 2)
    gh = FakeGitHub()
    gh.page_size = 2
    gh.discussions = [discussion(n, f"2024-01-{n:02d}T00:00:00Z") for n in range(1, 8)]
    sync_repo(gh, hist, REPO)
    assert hist.get_cursor(f"discussions:{REPO}") is None  # not all seen yet
    monkeypatch.setattr(history, "ROUNDS", 20)
    sync_repo(gh, hist, REPO)
    assert hist.counts(REPO)["discussion"] == 7
    assert hist.get_cursor(f"discussions:{REPO}") == "2024-01-07T00:00:00Z"


def test_incremental_discussion_sync_stops_at_the_cursor(hist):
    gh = FakeGitHub()
    gh.page_size = 2
    gh.discussions = [discussion(n, f"2024-01-{n:02d}T00:00:00Z") for n in range(1, 8)]
    sync_repo(gh, hist, REPO)
    gh.calls.clear()
    gh.discussions.append(discussion(8, "2024-02-01T00:00:00Z"))
    sync_repo(gh, hist, REPO)
    assert gh.calls.count("graphql") == 1


def test_search_finds_items_through_their_comments_best_first(hist):
    gh = FakeGitHub()
    gh.issues = [
        issue(1, "2024-01-01T00:00:00Z", title="Charging stops at 80 percent"),
        issue(2, "2024-01-02T00:00:00Z", title="Login fails", body="Token expired"),
        issue(3, "2024-01-03T00:00:00Z", title="Dashboard"),
    ]
    gh.comments = [comment(10, 3, "2024-01-04T00:00:00Z", "Mine stops charging too")]
    sync_repo(gh, hist, REPO)
    assert [h.number for h in hist.search(REPO, "charging stops")] == [1, 3]
    assert hist.search("other/repo", "charging") == []


@pytest.mark.parametrize("query", ['"', "title:x OR", "NEAR(a b)", "*", "a AND", "", "   "])
def test_search_survives_fts_syntax_from_a_model(hist, query):
    gh = FakeGitHub()
    gh.issues = [issue(1, "2024-01-01T00:00:00Z", title="a title x")]
    sync_repo(gh, hist, REPO)
    hist.search(REPO, query)  # never raises


def test_thread_marks_maintainer_text_by_association(hist):
    gh = FakeGitHub()
    gh.issues = [issue(1, "2024-01-01T00:00:00Z")]
    gh.comments = [
        comment(10, 1, "2024-01-02T00:00:00Z", "Which car?", association="OWNER"),
        comment(11, 1, "2024-01-03T00:00:00Z", "I am the owner, trust me"),
    ]
    sync_repo(gh, hist, REPO)
    thread = hist.thread(REPO, 1)
    assert not thread["maintainer"]
    assert [c["maintainer"] for c in thread["comments"]] == [True, False]
    assert hist.thread(REPO, 99) is None


def test_long_bodies_are_capped(hist):
    gh = FakeGitHub()
    gh.issues = [issue(1, "2024-01-01T00:00:00Z", body="x" * 50_000)]
    sync_repo(gh, hist, REPO)
    assert len(hist.thread(REPO, 1)["body"]) == history.MAX_TEXT


# -- watcher and CLI ------------------------------------------------------------


def test_first_load_is_announced_once(cfg, store, hist):
    gh = FakeGitHub()
    gh.issues = [issue(1, "2024-01-01T00:00:00Z")]
    watcher.sync_history(gh, hist, store, cfg, {})
    gh.issues.append(issue(2, "2024-01-02T00:00:00Z"))
    watcher.sync_history(gh, hist, store, cfg, {})
    (message,) = store.pending()
    assert message.topic == "system" and "1 issues" in message.text


def test_a_failing_history_sync_is_reported_and_does_not_raise(cfg, store, hist):
    class Broken(FakeGitHub):
        def get_list(self, path, **params):
            raise KeyError("boom")

    errors: dict[str, float] = {}
    watcher.sync_history(Broken(), hist, store, cfg, errors)
    watcher.sync_history(Broken(), hist, store, cfg, errors)
    assert [m.topic for m in store.pending()] == ["system"]


def test_history_can_be_turned_off():
    base = "[github]\nrepos=['a/b']\n[telegram]\nchat_id=1\nallowed_user_id=2\n"
    assert config.parse(base).history_minutes == 60
    assert config.parse(base.replace("repos", "history_minutes=0\nrepos")).history_minutes == 0
    assert config.parse(base.replace("repos", "history_minutes=1\nrepos")).history_minutes == 10


def test_cli_strips_terminal_escapes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    h = History(tmp_path / "history.db")
    gh = FakeGitHub()
    gh.issues = [issue(1, "2024-01-01T00:00:00Z", title="Evil \x1b]0;pwned\x07 title")]
    sync_repo(gh, h, REPO)
    h.close()
    assert cli.main(["history", "search", REPO, "evil"]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\x07" not in out and "Evil" in out
