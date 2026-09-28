"""Watchtower phase 1: config, polling, dedup, model-output guarding, rendering, gateway auth."""

from __future__ import annotations

import pytest

from watchtower import config, gateway, llm, render, watcher
from watchtower.events import Event, iso, poll_repo

NOW = 1_790_000_000.0
BEFORE = iso(NOW - 3600)
AFTER = iso(NOW + 60)


class FakeGitHub:
    def __init__(self) -> None:
        self.issues: list[dict] = []
        self.comments: list[dict] = []
        self.discussions: list[dict] = []
        self.calls: list[tuple[str, dict]] = []

    def get_list(self, path: str, **params: object) -> list[dict]:
        self.calls.append((path, params))
        items = self.comments if path.endswith("/comments") else self.issues
        return [i for i in items if i["updated_at"] >= params["since"]]

    def graphql(self, query: str, variables: dict) -> dict:
        return {"repository": {"discussions": {"nodes": self.discussions}}}


def issue(number: int, created: str, *, pr: bool = False, author: str = "stranger") -> dict:
    item = {
        "number": number,
        "title": f"Title {number}",
        "user": {"login": author},
        "body": "Body",
        "created_at": created,
        "updated_at": created,
        "html_url": f"https://github.com/owner/repo/{'pull' if pr else 'issues'}/{number}",
    }
    if pr:
        item["pull_request"] = {}
    return item


def comment(cid: int, number: int, created: str, *, pr: bool = False) -> dict:
    kind = "pull" if pr else "issues"
    return {
        "id": cid,
        "issue_url": f"https://api.github.com/repos/owner/repo/issues/{number}",
        "html_url": f"https://github.com/owner/repo/{kind}/{number}#issuecomment-{cid}",
        "user": {"login": "stranger"},
        "body": "Me too",
        "created_at": created,
        "updated_at": created,
    }


# -- config -----------------------------------------------------------------


def test_config_topics_default_to_general(cfg):
    assert cfg.topics == {"triage": 11, "reviews": None, "replies": None, "system": None}
    assert cfg.ignore_authors == {"owner"}


@pytest.mark.parametrize("model", ["gpt-oss:120b-cloud", "qwen3-coder:480b-cloud", "x:cloud"])
def test_config_refuses_cloud_models(model):
    text = f'[github]\nrepos=["a/b"]\n[telegram]\nchat_id=1\nallowed_user_id=2\n[llm]\nsummary_model="{model}"'
    with pytest.raises(ValueError, match="cloud"):
        config.parse(text)


def test_config_requires_telegram_ids():
    with pytest.raises(ValueError, match="allowed_user_id"):
        config.parse('[github]\nrepos=["a/b"]\n[telegram]\nchat_id=1')


# -- polling ----------------------------------------------------------------


def test_first_poll_ignores_everything_before_the_baseline(cfg, store):
    gh = FakeGitHub()
    gh.issues = [issue(1, BEFORE), issue(2, AFTER)]
    poll = poll_repo(gh, store, cfg, "owner/repo", {}, now=NOW)
    assert [e.key for e in poll.events] == ["owner/repo#2"]


def test_prs_route_to_reviews_and_issues_to_triage(cfg, store):
    gh = FakeGitHub()
    gh.issues = [issue(1, AFTER), issue(2, AFTER, pr=True)]
    gh.comments = [comment(7, 2, AFTER, pr=True)]
    poll = poll_repo(gh, store, cfg, "owner/repo", {}, now=NOW)
    assert [(e.kind, e.topic) for e in poll.events] == [
        ("issue", "triage"),
        ("pr", "reviews"),
        ("pr_comment", "reviews"),
    ]
    assert poll.events[2].title == "Title 2"


def test_ignored_authors_are_dropped_case_insensitively(cfg, store):
    gh = FakeGitHub()
    gh.issues = [issue(1, AFTER, author="OWNER")]
    assert poll_repo(gh, store, cfg, "owner/repo", {}, now=NOW).events == []


def test_cursor_is_returned_not_stored(cfg, store):
    gh = FakeGitHub()
    gh.issues = [issue(1, AFTER)]
    poll = poll_repo(gh, store, cfg, "owner/repo", {}, now=NOW)
    assert poll.cursors == {"issues:owner/repo": AFTER}
    assert store.get_cursor("issues:owner/repo") is None


def test_discussions_comments_and_replies(cfg, store):
    gh = FakeGitHub()
    reply = {"id": "R", "url": "u3", "createdAt": AFTER, "body": "", "author": None}
    gh.discussions = [
        {
            "number": 5,
            "title": "Idea",
            "url": "u1",
            "createdAt": BEFORE,
            "body": "",
            "author": {"login": "a"},
            "comments": {
                "nodes": [
                    {
                        "id": "C",
                        "url": "u2",
                        "createdAt": AFTER,
                        "body": "",
                        "author": {"login": "b"},
                        "replies": {"nodes": [reply]},
                    }
                ]
            },
        }
    ]
    poll = poll_repo(gh, store, cfg, "owner/repo", {}, now=NOW)
    assert [(e.kind, e.author) for e in poll.events] == [
        ("discussion_comment", "b"),
        ("discussion_comment", "ghost"),
    ]


def test_poll_once_reports_each_event_exactly_once(cfg, store, monkeypatch):
    monkeypatch.setattr(llm, "summarize", lambda cfg, item: None)
    gh = FakeGitHub()
    store.set_cursor("baseline:owner/repo", iso(NOW))
    gh.issues = [issue(1, AFTER)]
    for _ in range(3):
        watcher.poll_once(gh, store, cfg, {}, {})
    assert [m.text.splitlines()[1] for m in store.pending()] == ["Title 1"]
    assert store.get_cursor("issues:owner/repo") == AFTER


def test_a_failing_repo_is_reported_once_and_does_not_crash(cfg, store):
    class Broken(FakeGitHub):
        def get_list(self, path, **params):
            raise KeyError("boom")

    errors: dict[str, float] = {}
    watcher.poll_once(Broken(), store, cfg, {}, errors)
    watcher.poll_once(Broken(), store, cfg, {}, errors)
    assert [m.topic for m in store.pending()] == ["system"]


def test_bot_activity_is_silent_and_not_summarised(cfg, store, monkeypatch):
    def fail(cfg, item):
        raise AssertionError("bots must not reach the model")

    monkeypatch.setattr(llm, "summarize", fail)
    event = Event("k", "reviews", "pr", "owner/repo", 3, "Bump x", "dependabot[bot]", "", "u")
    watcher.handle(event, cfg, store)
    (message,) = store.pending()
    assert message.silent and store.is_seen("k")


# -- the model's answer is untrusted ----------------------------------------


def test_parse_accepts_the_schema_and_strips_thinking():
    content = '<think>hmm</think>{"kind": "bug", "summary": "SoC stuck", "needs_reply": true}'
    assert llm.parse(content) == llm.Summary("bug", "SoC stuck", True)


@pytest.mark.parametrize(
    "content",
    [
        "no json",
        '{"kind": "exploit", "summary": "x", "needs_reply": false}',
        '{"kind": "bug", "summary": 5, "needs_reply": false}',
        '["kind"]',
    ],
)
def test_parse_rejects_anything_else(content):
    assert llm.parse(content) is None


def test_parse_removes_links_and_mentions_and_caps_length():
    long = "x" * 500
    content = (
        '{"kind": "other", "summary": "see https://evil.example/a and www.x.io, ping @JustChr '
        + long
        + '", "needs_reply": false}'
    )
    summary = llm.parse(content)
    assert "evil" not in summary.text and "www." not in summary.text
    assert "@" not in summary.text
    assert len(summary.text) == llm.MAX_SUMMARY


# -- rendering --------------------------------------------------------------


def test_everything_from_outside_is_escaped():
    event = Event(
        "k", "triage", "issue", "owner/repo", 1, "<b>hi</b>", "a&b", "<a href=x>y</a>", "u"
    )
    text = render.message(event, llm.Summary("bug", "<i>x</i>", False))
    assert "<b>hi</b>" not in text and "&lt;b&gt;hi&lt;/b&gt;" in text
    assert "a&amp;b" in text and "&lt;i&gt;x&lt;/i&gt;" in text


def test_snippet_drops_template_comments_and_tags():
    assert render.snippet("<!-- fill this in -->\n\nReal   text") == "Real text"
    assert render.snippet('See <img width="3" alt="x"> here, 1 < 2') == "See here, 1 < 2"


# -- gateway ----------------------------------------------------------------


def _update(chat: int, user: int, text: str) -> dict:
    return {"update_id": 1, "message": {"chat": {"id": chat}, "from": {"id": user}, "text": text}}


@pytest.mark.parametrize(
    ("chat", "user", "allowed"),
    [(-1001, 42, True), (-1001, 7, False), (-5, 42, False), (42, 42, False)],
)
def test_only_the_allowed_user_in_the_group_is_heard(cfg, chat, user, allowed):
    assert (gateway.authorized(_update(chat, user, "/status"), cfg) is not None) is allowed


def test_command_strips_the_bot_name():
    assert gateway.command({"text": "/Status@my_bot extra"}) == "/status"
    assert gateway.command({"text": "hello"}) is None


def test_deliver_marks_permanent_failures_and_routes_topics(cfg, store, monkeypatch):
    from watchtower.telegram import TelegramError

    monkeypatch.setattr(gateway, "SEND_INTERVAL", 0)
    sent = []

    class FakeBot:
        def send(self, chat_id, text, *, thread_id=None, url=None, silent=False):
            if text == "bad":
                raise TelegramError(400, "Bad Request: message thread not found")
            sent.append((chat_id, thread_id, text))

    store.enqueue("triage", "ok")
    store.enqueue("system", "bad")
    gateway.deliver(FakeBot(), store, cfg)
    assert sent == [(-1001, 11, "ok")]
    assert store.pending() == []
    assert store.outbox_stats(0) == {"pending": 0, "failed": 1, "sent": 1}
