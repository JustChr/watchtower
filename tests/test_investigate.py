"""Investigating before judging: read-only tools, the bounded loop, code at the author's version."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from tests.test_analysis import FORM, RELEASES
from tests.test_drafts import (
    ThreadSource,
    add_draft,
    put_comment,
    put_issue,
    verdict,
)
from tests.test_knowledge import make_tarball
from watchtower import analysis, config, drafts, evaluate, investigate, llm, snapshot, watcher
from watchtower.history import History

REPO = "owner/repo"
COORDINATOR = "\n".join(
    [
        "class Coordinator:",
        "    def refresh(self):",
        "        for car in self.cars[:1]:  # (a+b)* is not a regex here",
        "            self.fetch(car)",
    ]
)


@pytest.fixture
def hist(tmp_path):
    h = History(tmp_path / "history.db")
    yield h
    h.close()


@pytest.fixture
def agent_cfg(cfg):
    return dataclasses.replace(cfg, agent_model="big:120b")


@pytest.fixture
def model(monkeypatch):
    from tests.test_drafts import FakeModel

    fake = FakeModel()
    monkeypatch.setattr(llm, "converse", fake)
    monkeypatch.setattr(llm, "act", fake.act)
    return fake


@pytest.fixture
def code(tmp_path):
    root = tmp_path / "code"
    (root / "custom_components" / "car").mkdir(parents=True)
    (root / "custom_components" / "car" / "coordinator.py").write_text(COORDINATOR, newline="")
    (root / "custom_components" / "car" / "blob.bin").write_bytes(b"\0refresh\0")
    (root / "node_modules").mkdir()
    (root / "node_modules" / "dep.js").write_text("refresh()")
    (tmp_path / "secret.txt").write_text("outside")
    return root


def workspace(hist, code=None, **extra) -> investigate.Workspace:
    return investigate.Workspace(hist, REPO, 7, code, "v1.0, the author's version", **extra)


def call(name: str, **args) -> dict:
    return {"function": {"name": name, "arguments": args}}


# -- the tools ----------------------------------------------------------------------


def test_code_tools_stay_inside_the_copy(hist, code, tmp_path):
    ws = workspace(hist, code)
    for path in (
        "../secret.txt",
        str(tmp_path / "secret.txt"),
        "/etc/passwd",
        "node_modules/dep.js",
    ):
        assert ws.run("read_code", {"path": path}).startswith("No file")
    assert ws.run("list_files", {"folder": ".."}).startswith("No folder")
    assert (
        ws.run("list_files", {})
        == "custom_components/car/blob.bin\ncustom_components/car/coordinator.py"
    )
    assert ws.read == {}


def test_a_link_out_of_the_copy_is_not_followed(hist, code, tmp_path):
    try:
        os.symlink(tmp_path / "secret.txt", code / "link.txt")
    except OSError:
        pytest.skip("no symlinks here")
    ws = workspace(hist, code)
    assert ws.run("read_code", {"path": "link.txt"}).startswith("No file")
    assert ws.run("search_code", {"text": "outside"}) == "No line contains «outside»."


def test_search_code_is_plain_text_and_remembers_what_it_found(hist, code):
    ws = workspace(hist, code)
    result = ws.run("search_code", {"text": "(A+B)*"})
    assert result == (
        "custom_components/car/coordinator.py:3:"
        "         for car in self.cars[:1]:  # (a+b)* is not a regex here"
    )
    assert ws.read == {"custom_components/car/coordinator.py": COORDINATOR}
    assert ws.run("search_code", {"text": "refresh", "folder": "custom_components"}) == (
        "custom_components/car/coordinator.py:2:     def refresh(self):"
    )
    assert ws.run("search_code", {"text": "nowhere"}) == "No line contains «nowhere»."
    assert ws.run("search_code", {"text": " "}) == "Give a text to search for."
    assert ws.steps[0] == "searched the code for «(A+B)*»"


def test_long_lines_and_many_hits_are_cut(hist, code):
    (code / "big.json").write_text("x" * 5000 + "needle" + "y" * 5000 + "\n" + "needle\n" * 100)
    ws = workspace(hist, code)
    result = ws.run("search_code", {"text": "needle"})
    first = result.splitlines()[0]
    assert first.startswith("big.json:1: …x") and first.endswith("y…")
    assert len(first) < 2 * investigate.SNIPPET + 30
    assert result.endswith(f"({101 - investigate.MAX_HITS} more hits not shown)")
    assert len(ws.run("read_code", {"path": "big.json"})) <= investigate.MAX_RESULT + 60


def test_read_code_numbers_and_clamps_lines(hist, code):
    ws = workspace(hist, code)
    path = "custom_components/car/coordinator.py"
    assert ws.run("read_code", {"path": path, "start_line": "2", "end_line": 3}) == (
        f"{path}, lines 2-3 of 4:\n2:     def refresh(self):\n"
        "3:         for car in self.cars[:1]:  # (a+b)* is not a regex here"
    )
    assert "lines 1-4" in ws.run("read_code", {"path": path, "start_line": -5, "end_line": True})
    (code / "long.py").write_text("\n".join(f"line {n}" for n in range(1, 1001)))
    assert "lines 10-209 of 1000" in ws.run(
        "read_code", {"path": "long.py", "start_line": 10, "end_line": 900}
    )
    assert ws.run("read_code", {"path": "custom_components/car/blob.bin"}).endswith(
        "not a text file."
    )
    assert ws.steps[-1] == "read long.py:10-209"


def test_attachment_tools(hist):
    diagnostics = json.dumps({"data": {"lock.status": "LOCKED", "doors": ["open"]}}, indent=2)
    ws = workspace(hist, files={"config_entry-car.json": diagnostics})
    assert ws.run("search_attachment", {"name": "CONFIG_ENTRY-car.json", "text": "lock"}) == (
        '3:     "lock.status": "LOCKED",'
    )
    assert ws.run("read_attachment", {"name": "config_entry-car.json", "end_line": 2}).endswith(
        '1: {\n2:   "data": {'
    )
    assert ws.run("read_attachment", {"name": "x.log"}) == (
        "No attached file 'x.log'. Attached: config_entry-car.json."
    )
    assert "search_code" not in json.dumps(ws.tools())  # no code: no code tools
    assert ws.run("search_code", {"text": "x"}).startswith("There is no tool 'search_code'.")


def test_threads_are_searched_as_they_were_then(hist):
    put_issue(hist, 7, title="Only one car refreshed")
    put_issue(hist, 3, title="Second car never refreshed", state="completed")
    put_comment(hist, 30, 3, "The car list is cut.", association="OWNER")
    put_comment(hist, 31, 3, "Fixed in v2.", association="OWNER", created="2026-10-01T00:00:00Z")
    put_issue(hist, 9, title="Car refreshed twice", created="2026-10-02T00:00:00Z")
    ws = workspace(hist, as_of="2026-09-15T00:00:00Z")
    assert ws.run("search_threads", {"words": "car refreshed"}) == (
        "#3 [issue, unknown] Second car never refreshed"
    )
    text = ws.run("read_thread", {"number": "3"})
    assert "The car list is cut." in text and "(maintainer)" in text and "Fixed in v2" not in text
    assert "#3" in ws.read and "Fixed in v2" not in ws.read["#3"]
    assert ws.run("read_thread", {"number": 7}) == "No earlier thread #7."  # itself
    assert ws.run("read_thread", {"number": 9}) == "No earlier thread #9."  # later


def test_a_failing_tool_or_bad_arguments_never_raise(hist, code, monkeypatch):
    ws = workspace(hist, code)
    assert ws.run(None, None).startswith("There is no tool ''")
    assert ws.run("read_code", "not a dict").startswith("No file")

    def broken(*args):
        raise PermissionError

    monkeypatch.setattr(investigate, "_text_file", broken)
    assert ws.run("search_code", {"text": "x"}) == "The tool failed (PermissionError)."


# -- the loop --------------------------------------------------------------------------


def start() -> list[dict]:
    return [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_the_loop_runs_tool_calls_until_the_model_stops(agent_cfg, hist, code, model):
    model.turns = [
        {"content": "", "thinking": "look", "tool_calls": [call("search_code", text="refresh")]},
        {
            "content": "",
            "thinking": "read",
            "tool_calls": [
                {"function": {"name": "read_code", "arguments": '{"path": "nope.py"}'}},
                call("list_files"),
            ],
        },
        {"content": "It fetches only the first car: coordinator.py:3."},
    ]
    messages = start()
    rounds = investigate.run(agent_cfg, messages, workspace(hist, code), 100_000)
    assert rounds == 2 and len(model.acts) == 3
    assert [m["role"] for m in messages] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "tool", "assistant",
    ]  # fmt: skip
    assert (
        messages[3]["tool_name"] == "search_code" and "coordinator.py:2" in messages[3]["content"]
    )
    assert messages[5]["content"].startswith("No file 'nope.py'")
    assert "thinking" not in messages[2] and "thinking" not in messages[4]  # only the latest
    assert "thinking" in model.acts[1][2]  # but the model saw its last thinking


def test_the_loop_is_bounded(agent_cfg, hist, code, model):
    cfg = dataclasses.replace(agent_cfg, agent_steps=3)
    model.turns = [{"tool_calls": [call("list_files")] * 7} for _ in range(10)]
    messages = start()
    assert investigate.run(cfg, messages, workspace(hist, code), 100_000) == 3
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert len(results) == 21
    assert results[6] == f"Not run: at most {investigate.MAX_CALLS} tool calls at a time."


def test_old_tool_results_are_dropped_when_the_context_fills(agent_cfg, hist, code, model):
    model.turns = [
        {"tool_calls": [call("read_code", path="custom_components/car/coordinator.py")]}
    ] * 4
    messages = start()
    limit = 900
    investigate.run(agent_cfg, messages, workspace(hist, code), limit)
    contents = [m["content"] for m in messages if m["role"] == "tool"]
    assert contents[0] == investigate.DROPPED and contents[-1] != investigate.DROPPED
    assert investigate.fit(messages, limit)
    assert not investigate.fit(start() + [{"role": "user", "content": "x" * 1000}], limit)


def test_a_repeated_call_is_answered_without_running_it_again(agent_cfg, hist, code, model):
    search = call("search_code", text="refresh")
    model.turns = [{"tool_calls": [search]}, {"tool_calls": [search, call("list_files")]}]
    messages = start()
    ws = workspace(hist, code)
    investigate.run(agent_cfg, messages, ws, 100_000)
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert results[1] == investigate.REPEATED and results[2].startswith("custom_components/")
    assert ws.steps == ["searched the code for «refresh»", "listed the files"]


def test_the_same_lines_asked_for_in_other_words_count_as_a_repeat(agent_cfg, hist, code, model):
    path = "custom_components/car/coordinator.py"
    model.turns = [
        {"tool_calls": [call("read_code", path=path)]},
        {"tool_calls": [call("read_code", path=path, start_line=1, end_line=900)]},
    ]
    messages = start()
    ws = workspace(hist, code)
    investigate.run(agent_cfg, messages, ws, 100_000)
    assert [m["content"] for m in messages if m["role"] == "tool"][1] == investigate.REPEATED
    assert ws.steps == [f"read {path}:1-4"]


def test_a_dropped_result_can_be_asked_for_again(agent_cfg, hist, code, model):
    read = call("read_code", path="custom_components/car/coordinator.py")
    model.turns = [{"tool_calls": [read]}, {"tool_calls": [call("list_files")]}] * 3
    messages = start()
    investigate.run(agent_cfg, messages, workspace(hist, code), 900)
    results = [m["content"] for m in messages if m["role"] == "tool"]
    assert investigate.DROPPED in results and investigate.REPEATED not in results


def test_a_failing_step_ends_the_investigation(agent_cfg, hist, code, model):
    model.fail = "act"
    messages = start()
    assert investigate.run(agent_cfg, messages, workspace(hist, code), 100_000) == 0
    assert messages == start()


def test_unknown_paths_are_the_ones_the_code_lacks(code):
    text = (
        "custom_components/car/coordinator.py:3 cuts the list; also sensor.py, see"
        " coordinator.py and e.g. v1.2 or custom_components/car/missing.py"
    )
    assert investigate.unknown_paths(text, code) == [
        "sensor.py",
        "custom_components/car/missing.py",
    ]
    assert investigate.unknown_paths(text, None) == []


# -- in a draft ------------------------------------------------------------------------------


def copy_at(tmp_path, tag: str, files: dict[str, str]):
    target = snapshot.version_path(tmp_path, REPO, tag)
    for name, text in files.items():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(text)
    return target


def test_a_draft_investigates_the_authors_version(agent_cfg, store, hist, tmp_path, model):
    put_issue(hist, body=FORM)
    hist.put_releases(REPO, RELEASES)
    copy_at(tmp_path, "v0.9.14-beta.3", {"custom_components/car/coordinator.py": COORDINATOR})
    main = snapshot.path_for(tmp_path, REPO)
    main.mkdir(parents=True)
    (main / "fixed.py").write_text("fixed")
    model.turns = [
        {"tool_calls": [call("read_code", path="custom_components/car/coordinator.py")]},
        {"content": "Only the first car is refreshed."},
    ]
    model.verdicts = [
        verdict(
            category="our_bug",
            evidence=[
                {
                    "source": "custom_components/car/coordinator.py",
                    "quote": "for car in self.cars[:1]:",
                    "point": "only the first car",
                }
            ],
            code="custom_components/car/coordinator.py:3 and car/sensor.py",
        )
    ]
    add_draft(store)
    result = drafts.generate(agent_cfg, store.claim_draft(), hist, tmp_path)

    system = model.acts[0][0]["content"]
    assert "The code is the project at v0.9.14-beta.3, the author's version." in system
    assert "Before your final answer, investigate with the tools." in system
    # The final assessment: no tool turns (gpt-oss would keep calling tools), the
    # investigation folded into the question as text.
    ((system, question),) = [m for k, m in model.calls if k == "assess"]
    assert "Before your final answer" not in system["content"]
    assert question["content"].endswith(investigate.FINAL_ASK)
    assert (
        "===== What you found investigating (your tool calls and their results) =====\n"
        '>>> read_code {"path": "custom_components/car/coordinator.py"}\n'
        "custom_components/car/coordinator.py, lines 1-4 of 4:\n1: class Coordinator:"
    ) in question["content"]
    assert "Your notes: Only the first car is refreshed." in question["content"]
    assert result.verdict.evidence[0].verified  # the code it read is a source
    assert result.verdict.looked_at == ("read custom_components/car/coordinator.py:1-4",)
    assert result.verdict.unknown_paths == ("car/sensor.py",)
    again = analysis.Verdict.from_json(result.verdict.to_json())
    assert again == result.verdict


def test_a_draft_cut_off_after_investigating_resumes_there(agent_cfg, store, hist, tmp_path, model):
    put_issue(hist, body=FORM)
    hist.put_releases(REPO, RELEASES)
    copy_at(tmp_path, "v0.9.14-beta.3", {"custom_components/car/coordinator.py": COORDINATOR})
    model.turns = [
        {"tool_calls": [call("read_code", path="custom_components/car/coordinator.py")]},
        {"content": "Only the first car is refreshed."},
    ]
    model.verdicts = [
        verdict(
            category="our_bug",
            evidence=[
                {
                    "source": "custom_components/car/coordinator.py",
                    "quote": "for car in self.cars[:1]:",
                    "point": "only the first car",
                }
            ],
        )
    ]
    add_draft(store)
    draft = store.claim_draft()
    stages = drafts.Stages()
    model.fail = "assess"  # e.g. the box restarted while the model was judging
    assert drafts.generate(agent_cfg, draft, hist, tmp_path, stages=stages) is None
    assert sorted(stages.done) == [drafts.FILES, drafts.INVESTIGATION]

    model.fail = None
    model.acts.clear()
    result = drafts.generate(agent_cfg, draft, hist, tmp_path, stages=drafts.Stages(stages.done))
    assert model.acts == []  # not investigated again
    # What it read was read again from the copy: its quote still checks out.
    assert result.verdict.evidence[0].verified
    assert result.verdict.looked_at == ("read custom_components/car/coordinator.py:1-4",)
    assert set(result.seconds) == {"files", "investigation", "assessment", "reply"}


def test_restore_keeps_to_the_code_copy(hist, code):
    ws = workspace(hist, code)
    ws.restore(
        ["../secret.txt", "node_modules/dep.js", "#99", "custom_components/car/coordinator.py"]
    )
    assert list(ws.read) == ["custom_components/car/coordinator.py"]


def test_without_the_authors_version_live_drafts_use_the_default_branch(
    agent_cfg, store, hist, tmp_path, model
):
    put_issue(hist)
    snapshot.path_for(tmp_path, REPO).mkdir(parents=True)
    add_draft(store)
    drafts.generate(agent_cfg, store.claim_draft(), hist, tmp_path)
    assert "at the default branch as it is today." in model.acts[0][0]["content"]


def test_a_replay_never_investigates_todays_code(agent_cfg, store, hist, tmp_path, model):
    put_issue(hist, body=FORM)
    hist.put_releases(REPO, RELEASES)
    snapshot.path_for(tmp_path, REPO).mkdir(parents=True)  # main: holds the fix
    add_draft(store)
    draft = store.claim_draft()
    drafts.generate(agent_cfg, draft, hist, tmp_path, as_of="2026-09-28T06:00:00Z")
    system = model.acts[0][0]["content"]
    assert investigate.NO_CODE in system

    copy_at(tmp_path, "v0.9.13", {"a.py": "x"})  # the newest release then; the form's is later
    model.acts.clear()
    drafts.generate(agent_cfg, draft, hist, tmp_path, as_of="2026-09-27T16:00:00Z")
    assert (
        "at v0.9.13, the newest release when the thread was written." in model.acts[0][0]["content"]
    )


def test_agent_steps_zero_means_one_prompt_as_before(agent_cfg, store, hist, tmp_path, model):
    put_issue(hist)
    add_draft(store)
    cfg = dataclasses.replace(agent_cfg, agent_steps=0)
    result = drafts.generate(cfg, store.claim_draft(), hist, tmp_path)
    assert model.acts == [] and result.verdict.looked_at == ()
    (messages,) = [m for k, m in model.calls if k == "assess"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Before your final answer" not in messages[0]["content"]


def test_an_old_verdict_without_the_new_fields_still_loads():
    old = json.dumps(
        {
            "category": "our_bug",
            "confidence": "low",
            "evidence": [],
            "missing": [],
            "code": "",
            "fix": "",
            "attempts": 1,
        }
    )
    assert analysis.Verdict.from_json(old).looked_at == ()


def test_agent_steps_setting():
    from tests.conftest import CONFIG

    assert config.parse(CONFIG).agent_steps == 20
    assert config.parse(CONFIG + "agent_steps = -3\n").agent_steps == 0


# -- fetching the code at a version ------------------------------------------------------------


class TagSource:
    def __init__(self) -> None:
        self.downloads: list[str] = []

    def download(self, path, dest, max_bytes):
        self.downloads.append(path)
        dest.write_bytes(make_tarball({"a.py": path}))


def test_code_at_a_version_is_fetched_once_and_pruned(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "KEEP_VERSIONS", 2)
    source = TagSource()
    assert snapshot.fetch_version(source, REPO, "../../x", tmp_path) is None
    assert snapshot.fetch_version(source, REPO, "v1/2", tmp_path) is None
    first = snapshot.fetch_version(source, REPO, "v1.0", tmp_path)
    assert (first / "a.py").read_text() == f"/repos/{REPO}/tarball/v1.0"
    snapshot.fetch_version(source, REPO, "v1.0", tmp_path)
    assert source.downloads == [f"/repos/{REPO}/tarball/v1.0"]
    os.utime(first, (1, 1))
    snapshot.fetch_version(source, REPO, "v1.1", tmp_path)
    snapshot.fetch_version(source, REPO, "v1.2", tmp_path)
    assert sorted(p.name for p in first.parent.iterdir()) == ["v1.1", "v1.2"]


def test_fetch_code_takes_the_version_from_the_thread(hist, tmp_path):
    hist.put_releases(REPO, RELEASES)
    source = TagSource()
    thread = {"body": FORM, "comments": []}
    path = drafts.fetch_code(source, hist, REPO, thread, None, tmp_path)
    assert path == snapshot.version_path(tmp_path, REPO, "v0.9.14-beta.3") and path.is_dir()
    assert (
        drafts.fetch_code(source, hist, REPO, {"body": "", "comments": []}, None, tmp_path) is None
    )
    drafts.fetch_code(
        source, hist, REPO, {"body": "", "comments": []}, None, tmp_path, "2026-09-21"
    )
    assert source.downloads[-1] == f"/repos/{REPO}/tarball/v0.9.12"


def test_the_watcher_fetches_the_code_when_preparing_a_draft(store, hist, tmp_path, monkeypatch):
    monkeypatch.setattr("watchtower.attachments.download", lambda *a: 0)
    seen = []
    monkeypatch.setattr(
        drafts, "fetch_code", lambda source, h, repo, thread, files, root: seen.append(root)
    )
    add_draft(store, status="prep")
    watcher.prepare_drafts(ThreadSource(), hist, store, {}, tmp_path, tmp_path / "r")
    assert seen == [tmp_path / "r"]

    def broken(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(drafts, "fetch_code", broken)
    add_draft(store, "k2", status="prep")
    watcher.prepare_drafts(ThreadSource(), hist, store, {}, tmp_path, tmp_path)
    assert len(store.drafts("queued")) == 2  # a draft even without the code


def test_a_queued_replay_runs_offline_in_the_worker(agent_cfg, store, hist, tmp_path, monkeypatch):
    from watchtower.worker import Worker

    calls = []

    def fake_run(cfg, history, repo, root, folder, out, numbers, limit, say, source=None):
        say("#160: assessing")
        calls.append((repo, numbers, limit, out, source))
        return ["one outcome"]

    monkeypatch.setattr(evaluate, "run", fake_run)
    payload = {"repo": REPO, "numbers": [25, [160, 5]], "limit": None, "out": "/x/r.md"}
    store.add_job("eval", "eval:1", payload)
    Worker(agent_cfg, store, hist, tmp_path).step()
    assert calls == [(REPO, [25, (160, 5)], None, Path("/x/r.md"), None)]  # no GitHub
    (message,) = store.pending()
    assert "Replay of owner/repo done: 1 issue(s)" in message.text
    assert store.heartbeats()["worker"][1] == "eval:1: #160: assessing"


def test_the_replay_fetches_the_code_as_it_was(agent_cfg, hist, tmp_path, model, monkeypatch):
    put_issue(hist, 8, state="completed", created="2026-09-21T00:00:00Z")
    put_comment(hist, 80, 8, "Answer", association="OWNER", created="2026-09-22T00:00:00Z")
    hist.put_releases(REPO, RELEASES)
    monkeypatch.setattr("watchtower.attachments.download", lambda *a: 0)
    source = TagSource()
    evaluate.run(
        agent_cfg, hist, REPO, tmp_path, tmp_path / "f", tmp_path / "r.md", say=print, source=source
    )
    assert source.downloads == [f"/repos/{REPO}/tarball/v0.9.12"]
    assert "at v0.9.12, the newest release" in model.acts[0][0]["content"]
