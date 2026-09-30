"""Phase 3: reviewing a stranger's pull request -- trigger, facts, findings, posting."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import test_drafts
from test_drafts import FakeGitHubApi, pem
from test_pr import HEAD, PullSource

from watchtower import drafts, llm, pr, render, reviews, watcher, worker
from watchtower.events import Event
from watchtower.github_app import App
from watchtower.history import History

REPO = "owner/repo"
PR_URL = f"https://github.com/{REPO}/pull/7"


@pytest.fixture
def hist(tmp_path):
    h = History(tmp_path / "history.db")
    yield h
    h.close()


@pytest.fixture
def agent_cfg(cfg):
    return dataclasses.replace(cfg, agent_model="big:120b")


def pr_event(**extra) -> Event:
    fields = {
        "key": f"{REPO}#7",
        "topic": "reviews",
        "kind": "pr",
        "repo": REPO,
        "number": 7,
        "title": "Fix the thing",
        "author": "stranger",
        "body": "Fixes #3",
        "url": PR_URL,
        "association": "NONE",
    }
    return Event(**{**fields, **extra})


def summary(cfg, item, context=""):
    return llm.Summary("other", "A fix.", False)


def add_pr_draft(store, status="queued"):
    store.add_draft(
        f"{REPO}#7",
        repo=REPO,
        number=7,
        kind="pr",
        topic="reviews",
        title="Fix",
        url=PR_URL,
        status=status,
    )


# -- the trigger ------------------------------------------------------------------------


def test_a_strangers_new_pr_gets_a_review_draft(agent_cfg, store, hist, monkeypatch):
    monkeypatch.setattr(llm, "summarize", summary)  # says no reply needed: a PR gets one anyway
    watcher.handle(pr_event(), agent_cfg, store)
    worker.Worker(agent_cfg, store, hist, Path("none")).summaries()

    (draft,) = store.drafts("prep")
    assert (draft.kind, draft.topic, draft.number) == ("pr", "reviews", 7)
    assert "drafting a reply" in store.pending()[0].text


@pytest.mark.parametrize(
    "changes",
    [{"association": "OWNER"}, {"author": "dependabot[bot]"}, {"kind": "pr_comment"}],
)
def test_no_review_for_maintainers_bots_or_pr_comments(
    agent_cfg, store, hist, monkeypatch, changes
):
    monkeypatch.setattr(llm, "summarize", summary)
    watcher.handle(pr_event(**changes), agent_cfg, store)
    worker.Worker(agent_cfg, store, hist, Path("none")).summaries()
    assert store.drafts("prep") == []


def test_reviews_can_be_switched_off(agent_cfg, store, hist, monkeypatch):
    monkeypatch.setattr(llm, "summarize", summary)
    off = dataclasses.replace(agent_cfg, review_prs=False)
    watcher.handle(pr_event(), off, store)
    worker.Worker(off, store, hist, Path("none")).summaries()
    assert store.drafts("prep") == []
    assert not worker.wants_draft(pr_event(), None, off)


class PrSource(PullSource):
    """The PR's files and its thread, for the watcher's prep."""

    def get_json(self, path):
        if path == f"/repos/{REPO}/issues/7":
            return {
                "number": 7,
                "title": "Fix the thing",
                "user": {"login": "stranger"},
                "author_association": "NONE",
                "state": "open",
                "labels": [],
                "body": "Fixes #3",
                "created_at": "2026-09-01T00:00:00Z",
                "updated_at": "2026-09-03T00:00:00Z",
                "html_url": PR_URL,
                "pull_request": {},
            }
        return super().get_json(path)

    def get_list(self, path, **params):
        return []


def test_the_watcher_fetches_the_pr_for_a_review(store, hist, tmp_path):
    add_pr_draft(store, "prep")
    watcher.prepare_drafts(PrSource(), hist, store, {}, tmp_path / "files", tmp_path / "repos")

    assert [d.status for d in store.drafts("queued")] == ["queued"]
    data = pr.load(tmp_path / "repos", REPO, 7)
    assert data["head_sha"] == HEAD and data["code"] is True
    assert hist.thread(REPO, 7)["kind"] == "pr"


def test_a_pr_that_cannot_be_fetched_is_tried_again_then_fails(store, hist, tmp_path):
    class Down(PrSource):
        def get_json(self, path):
            if "/pulls/" in path:
                raise watcher.GitHubError("HTTP 502")
            return super().get_json(path)

    add_pr_draft(store, "prep")
    failures: dict[int, int] = {}
    for _ in range(watcher.PREP_TRIES):
        watcher.prepare_drafts(Down(), hist, store, failures, None, tmp_path / "repos")
    assert [d.status for d in store.drafts("failed")] == ["failed"]


# -- the facts code establishes ----------------------------------------------------------


def make_data(paths, body="", **extra):
    files = [
        {
            "path": p,
            "status": "modified",
            "previous": "",
            "additions": 1,
            "deletions": 0,
            "patch": "@@\n+x",
            "patch_cut": False,
        }
        for p in paths
    ]
    data = {
        "author": "stranger",
        "association": "NONE",
        "additions": 5,
        "deletions": 1,
        "commits": 1,
        "files": files,
        "changed_files": len(files),
        "body": body,
        "code": True,
        "head_sha": HEAD,
        "repo": REPO,
    }
    return data | extra


def test_facts_name_what_the_pr_lacks_and_touches(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text("# Changes")
    data = make_data(
        ["src/x.py", ".github/workflows/test.yml", "scripts/check.sh"],
        body="Fixes #3",
        mergeable=False,
        draft=True,
        head_repo="stranger/repo",
        code=False,
    )
    text = "\n".join(reviews.facts(data, tmp_path))

    assert "merge conflicts" in text and "marked as a draft" in text
    assert "From the fork stranger/repo" in text
    assert "changes code but touches no test file" in text
    assert "touches no docs" in text
    assert "keeps a changelog; this PR doesn't touch it" in text
    assert ".github/workflows/test.yml, scripts/check.sh" in text
    assert "It says it fixes #3" in text
    assert "couldn't be fetched: only the patches were read" in text


def test_facts_of_a_tidy_pr(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text("# Changes")
    paths = ["src/x.py", "tests/test_x.py", "CHANGELOG.md", "docs/a.md"]
    text = "\n".join(reviews.facts(make_data(paths), tmp_path))
    assert "Tests touched: tests/test_x.py" in text
    assert "no test file" not in text and "no docs" not in text
    assert "changelog" not in text.lower()


def test_a_linked_issue_is_looked_up_in_the_history(hist):
    hist.put_item(
        REPO,
        3,
        kind="issue",
        title="SoC stuck",
        author="a",
        association="NONE",
        state="open",
        labels="",
        body="",
        url="u",
        created="2026-09-01T00:00:00Z",
        updated="2026-09-01T00:00:00Z",
    )
    text = "\n".join(reviews.facts(make_data(["a.py"], body="closes #3"), None, hist))
    assert "It says it fixes #3 (issue, open): SoC stuck." in text


# -- the findings, checked ----------------------------------------------------------------


def answer(**extra) -> str:
    base = {
        "recommendation": "changes",
        "confidence": "medium",
        "summary": "It fixes the thing.",
        "findings": [],
        "missing": [],
        "decision": "",
    }
    return json.dumps(base | extra)


def finding(**extra) -> dict:
    base = {
        "severity": "blocker",
        "path": "src/f0.py",
        "quote": "x = 1",
        "point": "wrong",
        "fix": "use 2",
    }
    return base | extra


def code_dir(tmp_path):
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    (code / "src" / "f0.py").write_text("import os\n\nx = 1\ny = 2\n")
    (code / "other.txt").write_text("elsewhere")
    (tmp_path / "secret.txt").write_text("outside")
    return code


def test_a_finding_is_verified_against_the_patch_or_the_code_and_gets_its_line(tmp_path):
    data = make_data(["src/f0.py", "other.txt"])
    data["files"][0]["patch"] = "@@ -1 +1 @@\n-a\n+the patch line"
    got = reviews.parse_review(
        answer(
            findings=[
                finding(),  # in the code, line 3
                finding(quote="the patch line", point="in the patch"),  # patch only
                finding(quote="never written"),  # invented
                finding(path="other.txt", quote="elsewhere"),  # in that file's code, line 1
                finding(path="../secret.txt", quote="outside"),  # a path out of the copy
            ]
        ),
        data,
        code_dir(tmp_path),
    )
    marks = [(f.verified, f.line) for f in got.findings]
    assert marks == [(True, 3), (True, 0), (False, 0), (True, 1), (False, 0)]
    assert got.findings[0].where == "src/f0.py:3"
    assert [f.path for f in got.unverified] == ["src/f0.py", "../secret.txt"]


def test_a_quote_from_another_files_patch_does_not_verify_a_finding():
    data = make_data(["a.py", "b.py"])
    data["files"][1]["patch"] = "+only in b"
    got = reviews.parse_review(
        answer(findings=[finding(path="a.py", quote="only in b")]), data, None
    )
    assert got.findings[0].verified is False


def test_the_review_answer_is_parsed_against_a_fixed_shape():
    data = make_data(["a.py"])
    assert reviews.parse_review("not json", data, None) is None
    assert reviews.parse_review(answer(recommendation="approve!"), data, None) is None
    got = reviews.parse_review(
        answer(
            confidence="certain",
            summary="See https://evil.example and @everyone",
            findings=[finding(severity="catastrophic", point="x" * 900)] * 12,
            missing=["a test", "", "docs"],
        ),
        data,
        None,
    )
    assert got.confidence == "low"
    assert "evil.example" not in got.summary and "@" not in got.summary
    assert len(got.findings) == reviews.MAX_FINDINGS
    assert got.findings[0].severity == "should_fix" and len(got.findings[0].point) <= 400
    assert got.missing == ("a test", "docs")


def test_a_reviews_json_is_not_an_issues_verdict():
    from watchtower.analysis import Verdict

    v = reviews.ReviewVerdict("merge", "high", "s", (reviews.Finding("nit", "a", "q", "p"),))
    assert Verdict.from_json(v.to_json()) is None
    assert reviews.ReviewVerdict.from_json(v.to_json()) == v
    assert reviews.ReviewVerdict.from_json("") is None


# -- the whole review ---------------------------------------------------------------------


class ReviewModel:
    def __init__(self) -> None:
        self.answers: list[str] = []
        self.reply = {"reply": "Thanks! Please add a test.", "note": "check the test ask"}
        self.calls: list[tuple[str, list[dict]]] = []
        self.turns: list[dict] = []
        self.tool_names: list[str] = []

    def act(self, cfg, model, messages, tools, *, num_ctx, timeout, think=""):
        self.calls.append(("act", [dict(m) for m in messages]))
        self.tool_names = [t["function"]["name"] for t in tools]
        return self.turns.pop(0) if self.turns else {"content": "Nothing more."}

    def __call__(self, cfg, model, messages, *, num_ctx, timeout, schema=None, think=""):
        kind = {id(reviews.SCHEMA): "assess", id(drafts.SCHEMA): "reply"}[id(schema)]
        self.calls.append((kind, [dict(m) for m in messages]))
        if kind == "assess":
            return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return json.dumps(self.reply)


@pytest.fixture
def model(monkeypatch):
    fake = ReviewModel()
    monkeypatch.setattr(llm, "converse", fake)
    monkeypatch.setattr(llm, "act", fake.act)
    return fake


@pytest.fixture
def prepared(store, hist, tmp_path):
    """A PR the watcher fetched, the default branch's docs, and its claimed draft."""

    root = tmp_path / "repos"
    main = root / "owner" / "repo"
    (main / "docs" / "knowledge").mkdir(parents=True)
    (main / "CHANGELOG.md").write_text("# Changes")
    (main / "CLAUDE.md").write_text("Never edit generated files by hand.")
    (main / ".claude" / "skills" / "review-pr").mkdir(parents=True)
    (main / ".claude" / "skills" / "review-pr" / "SKILL.md").write_text("Check REST quota.")
    pr.fetch(PullSource(), REPO, 7, root)
    hist.put_item(
        REPO,
        7,
        kind="pr",
        title="Fix the thing",
        author="stranger",
        association="NONE",
        state="open",
        labels="",
        body="Fixes #3",
        url=PR_URL,
        created="2026-09-01T00:00:00Z",
        updated="2026-09-01T00:00:00Z",
    )
    add_pr_draft(store)
    return root, store.claim_draft()


def test_a_review_is_facts_investigation_assessment_then_text(agent_cfg, hist, model, prepared):
    root, draft = prepared
    model.answers = [
        answer(
            findings=[finding(path="src/f0.py", quote="+b", point="renames a")],
            missing=["a test"],
        )
    ]
    model.turns = [
        {"content": "", "tool_calls": [{"function": {"name": "list_changes", "arguments": {}}}]}
    ]
    result = reviews.generate(agent_cfg, draft, hist, root, stages=drafts.Stages())

    assert [k for k, _ in model.calls] == ["act", "act", "assess", "reply"]
    assert {"list_changes", "read_patch", "read_code", "search_docs"} <= set(model.tool_names)
    # the rules come from the default branch, the review skill included
    system = model.calls[0][1][0]["content"]
    assert "Never edit generated files by hand" in system and "Check REST quota" in system
    first_user = model.calls[0][1][1]["content"]
    assert "It changes code but touches no test file" in first_user
    assert "src/f0.py: modified, +3 -1" in first_user and "@@ -1 +1 @@" in first_user
    assert "src/f0.py: modified" in model.calls[1][1][-1]["content"]  # the tool's result
    # the writing pass is given the checked findings, and code closes the review
    reply_prompt = model.calls[-1][1][1]["content"]
    assert "[blocker, checked] src/f0.py" in reply_prompt and "Missing: a test" in reply_prompt
    assert result.reply == f"Thanks! Please add a test.\n\n_Reviewed at commit `{HEAD[:7]}`._"
    assert result.verdict.commit == HEAD
    assert result.verdict.judged_at.startswith("the pull request")
    assert result.verdict.findings[0].verified and "listed the changes" in result.verdict.looked_at


def test_unchecked_findings_go_back_to_the_model_once_more(agent_cfg, hist, model, prepared):
    root, draft = prepared
    model.answers = [
        answer(findings=[finding(quote="invented"), finding(quote="+b", point="real")]),
        answer(findings=[finding(quote="+b", point="real")]),
    ]
    result = reviews.generate(agent_cfg, draft, hist, root, stages=drafts.Stages())

    assert [k for k, _ in model.calls if k == "assess"] == ["assess", "assess"]
    correction = model.calls[-2][1][-1]["content"]
    assert '"invented"' in correction and '"+b"' not in correction
    assert result.verdict.attempts == 2 and not result.verdict.unverified


def test_no_review_without_the_fetched_pr_or_from_a_nonsense_answer(
    agent_cfg, hist, model, prepared, tmp_path
):
    root, draft = prepared
    assert reviews.generate(agent_cfg, draft, hist, tmp_path / "empty") is None
    model.answers = ["nonsense"]
    assert reviews.generate(agent_cfg, draft, hist, root, stages=drafts.Stages()) is None


def test_a_decision_the_review_leaves_is_named_and_blocks_posting(agent_cfg, hist, model, prepared):
    root, draft = prepared
    model.answers = [answer(decision="Rename the option (A) or keep it (B)")]
    result = reviews.generate(agent_cfg, draft, hist, root, stages=drafts.Stages())
    assert "may decide it for you" in result.note  # the text made no room for the decision

    model.reply = {"reply": "Thanks!\n[YOUR DECISION: …]", "note": "n"}
    result = reviews.generate(agent_cfg, draft, hist, root, stages=drafts.Stages())
    assert drafts.open_decision(result.reply)
    assert "Rename the option (A) or keep it (B)" in result.reply
    assert "Fill it in before posting" in result.note


def test_the_worker_reviews_a_pr_and_offers_it(agent_cfg, store, hist, model, prepared):
    root, draft = prepared
    store.requeue_drafting()  # the fixture claimed it; the worker claims it itself
    model.answers = [answer(findings=[finding(quote="+b")])]
    assert worker.Worker(agent_cfg, store, hist, root).step()

    messages = store.pending()
    assert any("Review of #7" in m.text and "🛑" in m.text for m in messages)
    offer = next(m for m in messages if "Draft review #7" in m.text)
    assert "never approves or merges" in offer.text
    assert [label for label, _ in offer.buttons] == ["✅ Post", "🗑 Reject"]  # no bug label
    assert not any("For Claude" in m.text for m in messages)


def test_a_long_review_assessment_fits_one_telegram_message():
    findings = tuple(
        reviews.Finding("blocker", "src/some/long/file.py", "q" * 150, "p" * 400, "f", True, 12)
        for _ in range(8)
    )
    v = reviews.ReviewVerdict(
        "changes",
        "high",
        "s" * 500,
        findings,
        ("m" * 300,) * 5,
        "d" * 400,
        ("f" * 300,) * 8,
        judged_at="the patches only",
        looked_at=("a",),
    )
    draft = type("D", (), {"repo": REPO, "number": 7, "title": "t" * 400})()
    text = render.review_verdict(draft, v)
    assert len(text) <= 4096 and "more (the web UI has them all)" in text


# -- posting -------------------------------------------------------------------------------


class ReviewApi(FakeGitHubApi):
    def __init__(self) -> None:
        super().__init__()
        self.permissions: dict = {}
        self.bodies: list[tuple[str, dict]] = []

    def __call__(self, token):
        client = super().__call__(token)
        api = self

        def post_json(path, body):
            api.calls.append((token[:3], "POST", path))
            if path.endswith("/access_tokens"):
                api.permissions = body["permissions"]
                return {"token": "ghs_x", "expires_at": "2026-09-28T13:00:00Z"}
            api.bodies.append((path, body))
            return {"html_url": f"https://github.com/{REPO}/pull/7#pullrequestreview-1"}

        client.post_json = post_json
        return client


key = test_drafts.key  # the signing-key fixture


def test_an_approved_review_is_posted_as_a_comment_review(key, store):
    api = ReviewApi()
    app = App("1", pem(key), connect=api, clock=lambda: 1_790_000_000.0)
    add_pr_draft(store)
    draft = store.draft(store.claim_draft().id)
    url = app.post(draft, "Looks fine, one test missing.")

    assert url.endswith("#pullrequestreview-1")
    assert api.permissions == {"pull_requests": "write"}  # issue comments don't ask for it
    assert api.bodies == [
        (
            f"/repos/{REPO}/pulls/7/reviews",
            {"body": "Looks fine, one test missing.", "event": "COMMENT"},
        )
    ]


# -- revising, and the web page -------------------------------------------------------------


def review_draft(store, hist, decision=""):
    """A ready review whose assessment is stored, as the worker leaves it."""

    add_pr_draft(store)
    claimed = store.claim_draft()
    verdict = reviews.ReviewVerdict(
        "changes",
        "high",
        "It fixes the thing.",
        (reviews.Finding("blocker", "src/f0.py", "x = 1", "wrong", "use 2", True, 3),),
        ("a test",),
        decision,
        ("By stranger.",),
        commit=HEAD,
        judged_at="the pull request's head ccccccc",
    )
    store.finish_draft(claimed.id, "Thanks!\n[YOUR DECISION: …]", "n", "", verdict.to_json())
    return store.draft(claimed.id)


def test_a_review_is_revised_from_its_own_assessment(agent_cfg, store, hist, model, tmp_path):
    draft = review_draft(store, hist, decision="Rename it (A) or keep it (B)")
    model.reply = {"reply": "Thanks!\n[YOUR DECISION: …]", "note": "Done."}
    got = drafts.revise(agent_cfg, draft, "Thanks!", "Shorter, please", hist, tmp_path)

    system, user = model.calls[-1][1][0]["content"], model.calls[-1][1][1]["content"]
    assert "review of a pull request" in system and "commit" in system
    assert "[blocker, checked] src/f0.py:3" in user and "Missing: a test" in user
    assert user.endswith("===== The maintainer's instruction =====\nShorter, please")
    assert got == ("Thanks!\n[YOUR DECISION: Rename it (A) or keep it (B)]", "Done.")


def test_the_web_page_shows_a_reviews_assessment(cfg, tmp_path, hist):
    from watchtower import web

    data = web.Data(cfg, tmp_path)
    draft = review_draft(data.store, data.history)
    (row,) = data.drafts()
    assert (row["category"], row["confidence"]) == ("changes", "high")
    found = data.draft(draft.id)["verdict"]
    assert found["review"] is True and found["label"] == "✏️ needs changes"
    assert found["findings"][0]["where"] == "src/f0.py:3" and found["findings"][0]["mark"] == "🛑"
    assert found["facts"] == ["By stranger."] and found["commit"] == HEAD
    assert data.draft(draft.id)["handoff"] is None  # no bug prompt for a PR
