"""The web UI and what it needs: the model-call trace, web-origin decisions, the HTTP guards."""

from __future__ import annotations

import http.client
import io
import json
import sqlite3
import threading
from http.server import ThreadingHTTPServer

import pytest

from tests.test_drafts import FakePoster, add_draft, put_issue, ready_draft
from watchtower import config, llm, poster, trace, web, worker
from watchtower.history import History
from watchtower.store import Store

# -- the trace ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ({"tools": [{"function": {"name": "read_code"}}]}, "investigate"),
        ({"format": {"properties": {"kind": {}, "needs_reply": {}}}}, "summary"),
        ({"format": {"properties": {"category": {}}}}, "assess"),
        ({"format": {"properties": {"findings": {}}}}, "findings"),
        ({"format": {"properties": {"reply": {}, "note": {}}}}, "reply"),
        ({}, "brief"),
    ],
)
def test_a_call_is_named_by_its_shape(payload, kind):
    assert trace.kind_of(payload) == kind


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_every_chat_call_is_traced_with_its_subject_and_step(cfg, tmp_path, monkeypatch):
    t = trace.Trace(tmp_path / "trace.db")
    monkeypatch.setattr(llm, "tracer", t.record)
    answers = [{"message": {"content": "hello", "thinking": "hm"}}, OSError("reset")]

    def fake_urlopen(request, timeout):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return FakeResponse(json.dumps(answer).encode())

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    t.subject, t.step = "draft:7", "writing the reply"
    assert llm.chat(cfg, "m", "sys", "user text", num_ctx=4096, timeout=5) == "hello"
    with pytest.raises(OSError):
        llm.chat(cfg, "m", "sys", "again", num_ctx=4096, timeout=5)

    failed, done = t.calls(subject="draft:7")  # newest first
    assert (done["step"], done["kind"], done["model"], done["num_ctx"]) == (
        "writing the reply",
        "brief",
        "m",
        4096,
    )
    assert done["prompt_chars"] == len("sys") + len("user text") and done["answer_chars"] == 7
    assert failed["error"] == "OSError: reset"
    full = t.call(done["id"])
    assert full["request"]["messages"][1]["content"] == "user text"
    assert full["answer"]["thinking"] == "hm"
    assert t.daily(0)[0]["calls"] == 2
    assert t.prune(now=done["at"] + (trace.KEEP_DAYS + 1) * 86400) == 2


def test_a_summary_between_draft_passes_gets_its_own_subject(agent_cfg, store, hist, tmp_path):
    t = trace.Trace(tmp_path / "trace.db")
    w = worker.Worker(agent_cfg, store, hist, tmp_path, trace=t)
    seen = []
    with w.about("draft:3"):
        w.between("assessing, attempt 1")
        with w.about("event:owner/repo#comment-9"):
            seen.append((t.subject, t.step))
        seen.append((t.subject, t.step))
    seen.append((t.subject, t.step))
    assert seen == [
        ("event:owner/repo#comment-9", ""),
        ("draft:3", "assessing, attempt 1"),
        ("", ""),
    ]


@pytest.fixture
def hist(tmp_path):
    h = History(tmp_path / "history.db")
    yield h
    h.close()


@pytest.fixture
def agent_cfg(cfg):
    import dataclasses

    return dataclasses.replace(cfg, agent_model="big:120b")


# -- web-origin decisions ----------------------------------------------------------------


def test_the_poster_never_posts_a_decision_from_the_web(cfg, store):
    _, version = ready_draft(store)
    store.record_decision("draft", "post", version.id, origin="web")
    fake = FakePoster()
    poster.apply_decisions(store, cfg, fake)
    assert fake.posts == [] and store.open_decisions("draft") == []
    assert store.drafts("ready")  # still waiting for your tap


def test_posting_from_the_web_offers_the_version_in_telegram(cfg, store):
    draft, version = ready_draft(store, "Thanks!")
    store.record_decision("draft", "offer", version.id, origin="web")
    poster.apply_decisions(store, cfg, FakePoster())
    (offered,) = store.pending()
    assert poster.WEB_OFFER in offered.text and "<pre>Thanks!</pre>" in offered.text
    assert offered.buttons[0] == ("✅ Post", f"draft:post:{version.id}")

    store.add_version(draft.id, "newer", "user")  # the offered version is outdated now
    store.record_decision("draft", "offer", version.id, origin="web")
    poster.apply_decisions(store, cfg, FakePoster())
    assert "edited since" in store.pending()[-1].text


def test_web_edits_and_rejections_are_applied_like_telegrams(cfg, store, tmp_path):
    draft, version = ready_draft(store)
    data = web.Data(cfg, tmp_path)
    data.store = store
    assert "new version" in data.edit(draft.id, "My version")
    poster.apply_decisions(store, cfg, FakePoster())
    assert store.latest_version(draft.id).text == "My version"
    data.reject(store.latest_version(draft.id).id)
    poster.apply_decisions(store, cfg, FakePoster())
    assert store.draft(draft.id).status == "rejected"
    assert {d["origin"] for d in store.decisions("draft", draft.id)} == {"web"}


@pytest.mark.parametrize(
    ("text", "error"),
    [("", "empty"), ("x" * 5000, "at most"), ("fine", "the draft is rejected")],
    ids=["empty", "too long", "not ready"],
)
def test_web_edits_are_checked(cfg, store, tmp_path, text, error):
    draft, _ = ready_draft(store)
    if error.startswith("the draft"):
        store.reject_draft(draft.id)
    data = web.Data(cfg, tmp_path)
    data.store = store
    with pytest.raises(ValueError, match=error):
        data.edit(draft.id, text)


def test_old_decision_tables_get_an_origin(tmp_path):
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute(
        "CREATE TABLE decision (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, action TEXT NOT NULL,"
        " ref INTEGER NOT NULL, at REAL NOT NULL, applied REAL, text TEXT)"
    )
    db.execute("INSERT INTO decision (kind, action, ref, at) VALUES ('draft', 'post', 1, 0)")
    db.commit()
    db.close()
    store = Store(tmp_path / "old.db")
    assert store.decision_origin(1) == "telegram"  # the gateway recorded those
    store.close()


# -- what the page reads -------------------------------------------------------------------


def test_a_drafts_page_keeps_edits_and_button_presses_apart(cfg, tmp_path):
    data = web.Data(cfg, tmp_path)
    put_issue(data.history)
    # Draft 1's first version is version 1: equal ids, different things.
    draft, version = ready_draft(data.store)
    assert draft.id == version.id == 1
    data.store.record_decision("draft", "reply", draft.id, "an edit")
    data.store.record_decision("draft", "post", version.id)
    data.trace.subject = "draft:1"
    data.trace.record({"model": "m", "messages": []}, {"message": {}}, 1.0, 2.0)
    found = data.draft(1)
    assert [(d["action"], d["text"]) for d in found["decisions"]] == [
        ("reply", "an edit"),
        ("post", None),
    ]
    assert [c["subject"] for c in found["calls"]] == ["draft:1"]
    assert found["thread"]["title"] == "SoC stuck at 80" and found["verdict"] is None
    assert data.draft(99) is None


def test_the_overview_says_what_is_in_flight(cfg, tmp_path):
    data = web.Data(cfg, tmp_path)
    add_draft(data.store, status="prep")
    data.store.add_job("summary", "k", {})
    data.store.beat("worker", "idle")
    o = data.overview()
    assert o["queued_jobs"] == {"summary": 1} and o["drafts"] == {"prep": 1}
    assert [d["status"] for d in o["in_flight"]] == ["prep"]
    assert o["heartbeats"]["worker"]["detail"] == "idle"
    assert "chat_id" not in json.dumps(o)  # no personal ids on the page


# -- HTTP ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("192.168.1.20:8080", True),
        ("[::1]:8080", True),
        ("localhost:8080", True),
        ("jarvis.home.arpa:8080", True),
        ("JARVIS.home.arpa", True),
        ("evil.example:8080", False),  # DNS rebinding
        ("", False),
    ],
)
def test_only_known_host_names_are_answered(host, allowed):
    assert web.allowed_host(host, frozenset({"jarvis.home.arpa"})) is allowed


@pytest.fixture
def server(cfg, tmp_path):
    import dataclasses

    handler = type(
        "H",
        (web.Handler,),
        {"cfg": dataclasses.replace(cfg, web_hosts=("box.lan",)), "data_dir": tmp_path},
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], tmp_path
    httpd.shutdown()


def request(port, method, path, *, host="box.lan", body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, body=body, headers={"Host": host, **(headers or {})})
    response = conn.getresponse()
    return response.status, dict(response.getheaders()), response.read()


def test_the_page_and_its_api_are_served_with_a_strict_policy(server):
    port, _ = server
    status, headers, body = request(port, "GET", "/")
    assert status == 200 and b"<title>Watchtower</title>" in body
    assert "script-src 'self'" in headers["Content-Security-Policy"]
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert request(port, "GET", "/api/overview")[0] == 200
    assert request(port, "GET", "/fonts/barlow-latin-400-normal.woff2")[1]["Content-Type"] == (
        "font/woff2"
    )
    assert request(port, "GET", "/fonts/../../web.py")[0] == 404
    assert request(port, "GET", "/web.py")[0] == 404
    assert request(port, "GET", "/api/drafts/999")[0] == 404


def test_another_site_cannot_use_a_lan_browser(server, cfg):
    port, data_dir = server
    assert request(port, "GET", "/api/overview", host="evil.example")[0] == 421
    store = Store(data_dir / "watchtower.db")
    draft, version = ready_draft(store)
    # A plain form post from another site: no custom header, no JSON.
    status, _, _ = request(
        port,
        "POST",
        f"/api/versions/{version.id}/reject",
        body=b"x=1",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert status == 403
    assert request(port, "OPTIONS", f"/api/versions/{version.id}/reject")[0] == 405
    assert store.open_decisions("draft") == []

    status, _, body = request(
        port,
        "POST",
        f"/api/versions/{version.id}/reject",
        body=b"{}",
        headers={"Content-Type": "application/json", "X-Watchtower": "1"},
    )
    assert status == 200 and b"Rejected" in body
    ((decision_id, action, _, _),) = store.open_decisions("draft")
    assert action == "reject" and store.decision_origin(decision_id) == "web"
    status, _, body = request(
        port,
        "POST",
        f"/api/drafts/{draft.id}/edit",
        body=json.dumps({"text": ""}).encode(),
        headers={"Content-Type": "application/json", "X-Watchtower": "1"},
    )
    assert status == 400 and b"empty" in body
    assert (
        request(
            port,
            "POST",
            "/api/drafts/1/post",
            body=b"{}",
            headers={"Content-Type": "application/json", "X-Watchtower": "1"},
        )[0]
        == 404
    )  # there is no way to post from the web
    store.close()


def test_the_web_section_of_the_config(cfg):
    text = (
        "[github]\nrepos=['a/b']\n[telegram]\nchat_id=1\nallowed_user_id=2\n"
        "[web]\nport=9000\nhosts=['Box.LAN']"
    )
    parsed = config.parse(text)
    assert (parsed.web_port, parsed.web_hosts) == (9000, ("box.lan",))
    assert (cfg.web_port, cfg.web_hosts) == (8080, ())
