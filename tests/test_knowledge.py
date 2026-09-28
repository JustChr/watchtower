"""What agents learn a repo from: code snapshot, trusted docs, the brief and its approval."""

from __future__ import annotations

import dataclasses
import io
import sqlite3
import tarfile
import urllib.request

import pytest

from watchtower import brief, config, gateway, llm, render, snapshot, watcher
from watchtower.github import GitHub, GitHubError
from watchtower.history import History
from watchtower.store import Store

REPO = "owner/repo"
SHA1 = "a" * 40
SHA2 = "b" * 40


def make_tarball(files: dict[str, str], top: str = "owner-repo-abc1234") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeSource:
    def __init__(self, sha: str, files: dict[str, str]) -> None:
        self.sha = sha
        self.archive = make_tarball(files)
        self.downloads: list[str] = []
        self.releases: list[dict] = []

    def get_json(self, path: str) -> dict | list:
        if path == f"/repos/{REPO}":
            return {"default_branch": "main"}
        if path.startswith(f"/repos/{REPO}/releases?"):
            return self.releases
        assert path == f"/repos/{REPO}/branches/main"
        return {"commit": {"sha": self.sha}}

    def download(self, path, dest, max_bytes):
        self.downloads.append(path)
        dest.write_bytes(self.archive)


@pytest.fixture
def hist(tmp_path):
    h = History(tmp_path / "history.db")
    yield h
    h.close()


@pytest.fixture
def agent_cfg(cfg):
    return dataclasses.replace(cfg, agent_model="big:120b")


# -- snapshot ---------------------------------------------------------------


def test_snapshot_unpacks_the_branch_once_per_commit(tmp_path, hist):
    root = tmp_path / "repos"
    source = FakeSource(SHA1, {"README.md": "v1", "old.py": ""})
    assert snapshot.sync(source, hist, REPO, root) == SHA1
    target = root / "owner" / "repo"
    assert (target / "README.md").read_text() == "v1"

    snapshot.sync(source, hist, REPO, root)
    assert len(source.downloads) == 1  # unchanged commit: no download

    source.sha, source.archive = SHA2, make_tarball({"README.md": "v2"})
    snapshot.sync(source, hist, REPO, root)
    assert (target / "README.md").read_text() == "v2"
    assert not (target / "old.py").exists()
    assert sorted(p.name for p in target.parent.iterdir()) == ["repo"]  # no leftovers


def test_snapshot_refuses_paths_outside_the_target_and_keeps_the_old_copy(tmp_path, hist):
    root = tmp_path / "repos"
    snapshot.sync(FakeSource(SHA1, {"README.md": "good"}), hist, REPO, root)
    evil = FakeSource(SHA2, {})
    evil.archive = make_tarball({"top/ok.md": "", "../../evil.txt": "x"}, top="")
    with pytest.raises(snapshot.SnapshotError):
        snapshot.sync(evil, hist, REPO, root)
    assert not (tmp_path / "evil.txt").exists()
    assert (root / "owner" / "repo" / "README.md").read_text() == "good"
    assert hist.get_cursor(f"snapshot:{REPO}") == SHA1


def test_snapshot_refuses_archive_bombs(tmp_path, hist, monkeypatch):
    monkeypatch.setattr(snapshot, "MAX_FILES", 2)
    source = FakeSource(SHA1, {"a": "", "b": "", "c": ""})
    with pytest.raises(snapshot.SnapshotError, match="too large"):
        snapshot.sync(source, hist, REPO, tmp_path)


def test_snapshot_refuses_odd_commit_ids(tmp_path, hist):
    with pytest.raises(snapshot.SnapshotError):
        snapshot.sync(FakeSource("../../x", {}), hist, REPO, tmp_path)


def test_download_stops_at_the_size_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: io.BytesIO(b"x" * 100))
    with pytest.raises(GitHubError, match="over 10 bytes"):
        GitHub("t").download("/repos/o/r/tarball/x", tmp_path / "a", max_bytes=10)


# -- trusted docs -------------------------------------------------------------


def write(root, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_trusted_docs_come_most_useful_first_within_budget(tmp_path):
    write(
        tmp_path,
        {
            "Readme.md": "readme",
            "CLAUDE.md": "rules",
            "docs/setup.md": "setup",
            ".claude/skills/triage/SKILL.md": "triage",
            "src/app.py": "code",
        },
    )
    docs = snapshot.trusted_docs(tmp_path)
    assert [name for name, _ in docs] == [
        "CLAUDE.md",
        "Readme.md",
        ".claude/skills/triage/SKILL.md",
        "docs/setup.md",
    ]
    assert snapshot.trusted_docs(tmp_path, budget=7) == [
        ("CLAUDE.md", "rules"),
        ("Readme.md", "re"),
    ]
    assert snapshot.trusted_docs(tmp_path / "missing") == []


def test_tree_skips_dependency_folders(tmp_path):
    write(tmp_path, {"a.py": "", "node_modules/x.js": "", ".git/HEAD": "", "pkg/b.py": ""})
    assert snapshot.tree(tmp_path) == ["a.py", "pkg/b.py"]


def test_readme_excerpt_is_plain_prose(tmp_path):
    write(
        tmp_path,
        {
            "README.md": "<!-- hidden -->\n![badge](https://img.shields.io/x)\n# Bavarian Data\n"
            "Reads [CarData](https://bmw.example/api) for @someone. <img src=x>" + " more" * 500
        },
    )
    text = snapshot.readme_excerpt(tmp_path)
    assert text.startswith("# Bavarian Data Reads CarData for someone.")
    assert "hidden" not in text and "http" not in text and "<" not in text
    assert len(text) <= 1500 and text.endswith("…")


# -- the brief ------------------------------------------------------------------


def test_brief_parse_cleans_and_caps():
    content = (
        "<think>plan</think>\n\nPurpose\nSee https://x.io, ask @JustChr\n\n\n\nEnd" + "y" * 5000
    )
    text = brief.parse(content)
    assert text.startswith("Purpose\nSee [link] ask JustChr\n\nEnd")
    assert len(text) == brief.MAX_BRIEF
    assert brief.parse("<think>only thinking</think>  ") is None


def release(tag: str, published: str, *, beta: bool = False, **extra) -> dict:
    return {
        "tag_name": tag,
        "prerelease": beta,
        "draft": False,
        "published_at": published,
        "body": f"Notes for {tag}",
        **extra,
    }


def test_releases_are_newest_first_without_drafts():
    source = FakeSource(SHA1, {})
    source.releases = [
        release("v1.0.0", "2026-01-01T00:00:00Z"),
        release("v1.1.0b1", "2026-02-01T00:00:00Z", beta=True),
        release("v2.0.0", "", draft=True),
    ]
    found = snapshot.releases(source, REPO)
    assert [(r.tag, r.label) for r in found] == [
        ("v1.1.0b1", "v1.1.0b1 (beta)"),
        ("v1.0.0", "v1.0.0 (stable)"),
    ]


def test_plan_picks_the_newest_release_or_falls_back_to_main():
    stable = snapshot.Release("v1.0.0", False, "2026-01-01T00:00:00Z", "")
    beta = snapshot.Release("v1.1.0b1", True, "2026-02-01T00:00:00Z", "")
    assert brief.plan([beta, stable], SHA1, include_betas=True).ref == "v1.1.0b1"
    assert brief.plan([beta, stable], SHA1, include_betas=False).ref == "v1.0.0"
    assert brief.plan([beta], SHA1, include_betas=False) == brief.Plan(SHA1, "main @ aaaaaaa", None)


def test_each_new_release_gets_a_brief_right_away(hist):
    now = 1_790_000_000.0
    v1 = brief.Plan("v1.0.0", "v1.0.0 (stable)", snapshot.Release("v1.0.0", False, "", ""))
    v2 = brief.Plan("v1.1.0b1", "v1.1.0b1 (beta)", snapshot.Release("v1.1.0b1", True, "", ""))
    assert brief.due(hist, REPO, v1, now)
    hist.add_brief(REPO, v1.ref, v1.label, "text")
    assert not brief.due(hist, REPO, v1, now + 30 * 86400)  # same release
    assert brief.due(hist, REPO, v2, now)  # new release: no waiting


def test_without_releases_main_gets_a_brief_at_most_weekly(hist):
    now = 1_790_000_000.0
    assert brief.due(hist, REPO, brief.plan([], SHA1, True), now)
    hist.add_brief(REPO, SHA1, "main", "text")
    hist.db.execute("UPDATE brief SET created = ?", (now,))
    assert not brief.due(hist, REPO, brief.plan([], SHA1, True), now + 30 * 86400)
    assert not brief.due(hist, REPO, brief.plan([], SHA2, True), now + 86400)  # too soon
    assert brief.due(hist, REPO, brief.plan([], SHA2, True), now + brief.REFRESH_DAYS * 86400)


def test_failed_brief_is_retried_after_a_day(hist):
    now = 1_790_000_000.0
    hist.add_brief(REPO, SHA1, "main", None)
    hist.db.execute("UPDATE brief SET created = ?", (now,))
    target = brief.plan([], SHA1, True)
    assert not brief.due(hist, REPO, target, now + 3600)
    assert brief.due(hist, REPO, target, now + brief.RETRY_FAILED)


def test_rendered_brief_fits_a_telegram_message():
    text = render.brief(REPO, "<b>v1</b> (beta)", "<&>" * 3000)
    assert len(text) < 4096 and "<&>" not in text and "<b>v1" not in text
    assert "&lt;&amp;&gt;" in text and text.endswith("…")


def test_generate_prompts_with_docs_files_and_releases(agent_cfg, tmp_path, monkeypatch):
    write(tmp_path, {"README.md": "Reads car data", "src/app.py": ""})
    seen = {}

    def fake_chat(cfg, model, system, user, **kwargs):
        seen.update(model=model, user=user, **kwargs)
        return "Purpose\nReads car data."

    monkeypatch.setattr(llm, "chat", fake_chat)
    beta = snapshot.Release("v1.1.0b1", True, "2026-02-01T00:00:00Z", "New: trips")
    stable = snapshot.Release("v1.0.0", False, "2026-01-01T00:00:00Z", "")
    text = brief.generate(agent_cfg, REPO, tmp_path, beta, [beta, stable])
    assert text == "Purpose\nReads car data."
    assert seen["model"] == "big:120b" and seen["num_ctx"] == agent_cfg.agent_num_ctx
    for part in ("src/app.py", "===== README.md =====", "New: trips", "- v1.0.0 (stable)"):
        assert part in seen["user"]
    assert "describes release v1.1.0b1 (beta)" in seen["user"]
    assert brief.generate(agent_cfg, REPO, tmp_path / "empty") is None  # no docs, no call


def test_project_context_prefers_the_approved_brief(tmp_path, hist):
    write(tmp_path / "owner" / "repo", {"README.md": "From the readme"})
    assert brief.project_context(hist, REPO, tmp_path) == "From the readme"
    brief_id = hist.add_brief(REPO, "v1", "v1 (stable)", "From the brief")
    assert brief.project_context(hist, REPO, tmp_path) == "From the readme"  # pending
    hist.decide_brief(brief_id, approved=True)
    assert brief.project_context(hist, REPO, tmp_path) == "From the brief"


def test_decided_or_failed_briefs_stay_decided(hist):
    failed = hist.add_brief(REPO, "v1", "v1", None)
    hist.decide_brief(failed, approved=True)
    rejected = hist.add_brief(REPO, "v2", "v2", "text")
    hist.decide_brief(rejected, approved=False)
    hist.decide_brief(rejected, approved=True)
    assert hist.approved_brief(REPO) is None


# -- the watcher end to end -------------------------------------------------------


def test_a_release_brief_is_written_from_the_release_tag(
    agent_cfg, store, hist, tmp_path, monkeypatch
):
    seen = []

    def fake_generate(cfg, repo, target, release=None, releases=()):
        seen.append(((target / "README.md").read_text(), release.tag, len(releases)))
        return "Purpose\nCar data."

    monkeypatch.setattr(brief, "generate", fake_generate)
    source = FakeSource(SHA1, {"README.md": "release docs"})
    source.releases = [release("v1.0.0", "2026-01-01T00:00:00Z")]
    watcher.sync_code(source, hist, store, agent_cfg, {}, tmp_path)

    assert seen == [("release docs", "v1.0.0", 1)]
    assert source.downloads[-1] == f"/repos/{REPO}/tarball/v1.0.0"
    assert not (tmp_path / ".release" / "owner" / "repo").exists()  # temporary copy removed
    (message,) = store.pending()
    assert "v1.0.0 (stable)" in message.text


def test_new_brief_goes_up_for_approval_and_is_used_once_approved(
    agent_cfg, store, hist, tmp_path, monkeypatch
):
    monkeypatch.setattr(brief, "generate", lambda cfg, repo, target: "Purpose\nCar data.")
    source = FakeSource(SHA1, {"README.md": "readme"})
    watcher.sync_code(source, hist, store, agent_cfg, {}, tmp_path)
    watcher.sync_code(source, hist, store, agent_cfg, {}, tmp_path)  # same commit: no second brief

    (message,) = store.pending()
    assert message.topic == "system" and "Car data." in message.text
    assert "main @ aaaaaaa" in message.text
    (approve, reject) = message.buttons
    assert approve[1].startswith("brief:approve:") and reject[1].startswith("brief:reject:")

    store.record_decision(*gateway.parse_press(approve[1]))
    watcher.apply_decisions(store, hist)
    watcher.apply_decisions(store, hist)
    assert store.open_decisions("brief") == []
    assert brief.project_context(hist, REPO, tmp_path) == "Purpose\nCar data."


def test_no_agent_model_means_no_brief(cfg, store, hist, tmp_path, monkeypatch):
    monkeypatch.setattr(brief, "generate", lambda *a: pytest.fail("no agent model configured"))
    watcher.sync_code(FakeSource(SHA1, {"README.md": ""}), hist, store, cfg, {}, tmp_path)
    assert store.pending() == []


def test_failed_brief_is_reported_without_buttons(agent_cfg, store, hist, tmp_path, monkeypatch):
    monkeypatch.setattr(brief, "generate", lambda cfg, repo, target: None)
    watcher.sync_code(FakeSource(SHA1, {"README.md": ""}), hist, store, agent_cfg, {}, tmp_path)
    (message,) = store.pending()
    assert "failed" in message.text and message.buttons == ()


def test_summaries_get_the_project_context(cfg, monkeypatch):
    seen = {}

    def fake_post(cfg, path, payload, timeout):
        seen["system"] = payload["messages"][0]["content"]
        return {"message": {"content": '{"kind": "bug", "summary": "x", "needs_reply": false}'}}

    monkeypatch.setattr(llm, "_post", fake_post)
    llm.summarize(cfg, {"body": "b"}, "It reads BMW CarData.")
    assert seen["system"].endswith("It reads BMW CarData.")
    llm.summarize(cfg, {"body": "b"})
    assert seen["system"] == llm.SYSTEM


def test_agent_model_must_be_local():
    text = (
        "[github]\nrepos=['a/b']\n[telegram]\nchat_id=1\nallowed_user_id=2\n"
        "[llm]\nagent_model='gpt-oss:120b-cloud'"
    )
    with pytest.raises(ValueError, match="agent_model"):
        config.parse(text)


# -- buttons: outbox, gateway ------------------------------------------------------


def test_outbox_keeps_buttons(store):
    store.enqueue("system", "t", buttons=[("Yes", "brief:approve:1")])
    store.enqueue("system", "plain")
    assert [m.buttons for m in store.pending()] == [(("Yes", "brief:approve:1"),), ()]


def test_store_adds_the_buttons_column_to_an_old_database(tmp_path):
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute(
        "CREATE TABLE outbox (id INTEGER PRIMARY KEY, topic TEXT NOT NULL, text TEXT NOT NULL,"
        " url TEXT, silent INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, sent REAL,"
        " attempts INTEGER NOT NULL DEFAULT 0, error TEXT)"
    )
    db.execute("INSERT INTO outbox (topic, text, created) VALUES ('system', 'queued before', 0)")
    db.commit()
    db.close()
    store = Store(tmp_path / "old.db")
    store.enqueue("system", "t", buttons=[("Yes", "brief:approve:1")])
    assert [m.buttons for m in store.pending()] == [(), (("Yes", "brief:approve:1"),)]
    store.close()
    Store(tmp_path / "old.db").close()  # a second open doesn't migrate again


def _press(chat: int, user: int, data: str) -> dict:
    return {
        "update_id": 1,
        "callback_query": {
            "id": "q1",
            "from": {"id": user},
            "data": data,
            "message": {"message_id": 99, "chat": {"id": chat}},
        },
    }


@pytest.mark.parametrize(
    ("chat", "user", "allowed"),
    [(-1001, 42, True), (-1001, 7, False), (-5, 42, False)],
)
def test_only_the_allowed_user_can_press_buttons(cfg, chat, user, allowed):
    update = _press(chat, user, "brief:approve:1")
    assert (gateway.authorized_press(update, cfg) is not None) is allowed


@pytest.mark.parametrize(
    "data",
    [None, "", "brief:approve", "brief:approve:x", "brief:delete:1", "post:approve:1", "a:b:c:d"],
)
def test_unknown_button_data_is_ignored(data):
    assert gateway.parse_press(data) is None


def test_press_records_confirms_and_removes_the_buttons(cfg, store):
    calls = []

    class FakeBot:
        def call(self, method, params=None, **kwargs):
            calls.append((method, params))

    gateway.press(FakeBot(), store, cfg, _press(-1001, 42, "brief:reject:5")["callback_query"])
    assert store.open_decisions("brief") == [(1, "reject", 5)]
    assert calls == [
        ("answerCallbackQuery", {"callback_query_id": "q1", "text": "Brief discarded"}),
        (
            "editMessageReplyMarkup",
            {"chat_id": -1001, "message_id": 99, "reply_markup": {"inline_keyboard": []}},
        ),
    ]

    calls.clear()
    gateway.press(FakeBot(), store, cfg, _press(-1001, 42, "junk")["callback_query"])
    assert calls == [("answerCallbackQuery", {"callback_query_id": "q1"})]
    assert len(store.open_decisions("brief")) == 1
