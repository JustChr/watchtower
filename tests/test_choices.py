"""A decision left to the maintainer comes with its options as buttons; choosing one settles
the text, and only then can it be posted."""

from __future__ import annotations

import dataclasses

import pytest

from tests.test_drafts import FakeModel, FakePoster, add_draft, put_comment, put_issue, verdict
from watchtower import (
    analysis,
    drafts,
    gateway,
    llm,
    poster,
    render,
    reviews,
    telegram,
    web,
    worker,
)
from watchtower.history import History

OPEN = "Thanks!\n[YOUR DECISION: A (keep dates naive) or B (store the zone)]"
SETTLED = "Thanks! We'll store the zone."
OPTIONS = ("keep the dates naive", "store them with their zone")


@pytest.fixture
def hist(tmp_path):
    h = History(tmp_path / "history.db")
    yield h
    h.close()


@pytest.fixture
def agent_cfg(cfg):
    return dataclasses.replace(cfg, agent_model="big:120b")


def draft_with(
    store, text=OPEN, options=OPTIONS, decision="keep the dates naive or store the zone", number=7
):
    """A ready issue draft whose assessment left a decision, as the worker stores it."""

    add_draft(store, f"owner/repo#{number}", number)
    claimed = store.claim_draft()
    v = analysis.Verdict(
        "question", "high", (), (), "", "", decision=decision, options=tuple(options)
    )
    version = store.finish_draft(claimed.id, text, "n", "", v.to_json())
    return store.draft(claimed.id), version


# -- the options -----------------------------------------------------------------------------


def test_options_come_only_with_a_decision_two_to_four_short_and_plain():
    parse = analysis.parse_options
    assert parse(["a", "b"], "which one?") == ("a", "b")
    assert parse(["a", "b"], "") == ()  # no decision, no options
    assert parse(["only one"], "which?") == ()
    assert parse("a b", "which?") == () and parse(None, "which?") == ()
    assert parse(["a", "a", "b", 5, "", "c", "d", "e"], "which?") == ("a", "b", "c", "d")
    (long, _) = parse(["x" * 200, "see https://evil.example @all"], "which?")
    assert len(long) <= analysis.MAX_OPTION
    assert parse(["a", "see https://evil.example @all"], "which?")[1] == "see [link] all"


def test_the_assessment_keeps_the_options_and_old_rows_have_none():
    got = analysis.parse_verdict(
        json_of(verdict(decision="A or B?", options=["keep", "change"])), {}
    )
    assert got.options == ("keep", "change")
    assert analysis.Verdict.from_json(got.to_json()).options == ("keep", "change")

    old = analysis.Verdict.from_json(
        json_of(verdict(evidence=[], missing=[], asks=[], decision="A or B?"))
    )
    assert old.options == ()  # a row from before options existed


def json_of(data: dict) -> str:
    import json

    return json.dumps(data)


def test_a_reviews_assessment_has_options_too():
    data = {
        "recommendation": "changes",
        "confidence": "high",
        "summary": "s",
        "findings": [],
        "missing": [],
        "decision": "rename or keep",
        "options": ["rename it", "keep it"],
    }
    got = reviews.parse_review(json_of(data), {"files": []}, None)
    assert got.options == ("rename it", "keep it")
    assert reviews.ReviewVerdict.from_json(got.to_json()).options == got.options
    assert "Option 2: keep it" in reviews.assessment_text(got)
    assert reviews.parse_review(json_of(data | {"decision": ""}), {"files": []}, None).options == ()


def test_the_model_is_asked_for_options_in_both_kinds_of_assessment():
    assert "options" in analysis.ASSESS_SCHEMA["required"]
    assert "options" in reviews.SCHEMA["required"]
    assert '"options"' in analysis.ASSESS_SYSTEM and '"options"' in reviews.REVIEW_SYSTEM


# -- Telegram --------------------------------------------------------------------------------


def test_a_draft_with_an_open_decision_offers_its_options_instead_of_post(store):
    draft, version = draft_with(store)
    drafts.offer(store, draft, version)

    (message,) = store.pending()
    assert message.buttons == (
        ("1. keep the dates naive", f"draft:opt1:{version.id}"),
        ("2. store them with their zone", f"draft:opt2:{version.id}"),
        ("🗑 Reject", f"draft:reject:{version.id}"),
    )
    assert "Post" not in " ".join(label for label, _ in message.buttons)
    assert "A decision is left open" in message.text and "✅ Post comes once" in message.text


def test_a_settled_draft_gets_its_post_button_back(store):
    draft, version = draft_with(store, SETTLED)
    drafts.offer(store, draft, version)
    (message,) = store.pending()
    assert [label for label, _ in message.buttons] == ["✅ Post", "🗑 Reject"]
    assert "decision is left open" not in message.text


def test_an_open_decision_without_options_offers_only_reject(store):
    draft, version = draft_with(store, options=())
    drafts.offer(store, draft, version)
    (message,) = store.pending()
    assert [label for label, _ in message.buttons] == ["🗑 Reject"]  # reply with the decision


def test_a_reviews_open_decision_offers_its_options_too(store):
    store.add_draft(
        "k", repo="owner/repo", number=7, kind="pr", topic="reviews", title="t", url="u"
    )
    claimed = store.claim_draft()
    v = reviews.ReviewVerdict("changes", "high", "s", (), decision="rename?", options=("yes", "no"))
    version = store.finish_draft(claimed.id, OPEN, "n", "", v.to_json())
    drafts.offer(store, store.draft(claimed.id), version)
    (message,) = store.pending()
    assert [label for label, _ in message.buttons] == ["1. yes", "2. no", "🗑 Reject"]


def test_option_buttons_are_known_to_the_gateway_and_no_others():
    assert gateway.parse_press("draft:opt1:12") == ("draft", "opt1", 12)
    assert gateway.parse_press("draft:opt4:12") == ("draft", "opt4", 12)
    assert (
        gateway.parse_press("draft:opt5:12") is None
        and gateway.parse_press("draft:opt0:12") is None
    )
    assert drafts.MAX_CHOICES == 4


def test_long_option_buttons_get_a_row_each_and_short_ones_share(monkeypatch):
    sent = []
    bot = telegram.Bot("token")
    monkeypatch.setattr(bot, "call", lambda method, params, **kw: sent.append(params))

    bot.send(1, "x", buttons=(("✅ Post", "draft:post:1"), ("🗑 Reject", "draft:reject:1")))
    bot.send(
        1,
        "x",
        buttons=(("1. keep the dates naive", "a"), ("2. store them with their zone", "b")),
    )
    short, long = (m["reply_markup"]["inline_keyboard"] for m in sent)
    assert [len(row) for row in short] == [2]
    assert [len(row) for row in long] == [1, 1]


# -- a tap settles it ----------------------------------------------------------------------------


def press(store, cfg, version_id, number):
    store.record_decision("draft", f"opt{number}", version_id)
    poster.apply_decisions(store, cfg, FakePoster())


def test_a_tap_becomes_an_instruction_taken_from_the_assessment_not_the_button(cfg, store):
    draft, version = draft_with(store)
    press(store, cfg, version.id, 2)

    (job,) = store.jobs("queued")
    assert job.kind == "revise" and job.payload["draft"] == draft.id
    assert job.payload["instruction"] == drafts.choose_instruction(2, OPTIONS[1])
    assert "option 2, store them with their zone" in job.payload["instruction"]
    assert store.open_decisions("draft") == []


@pytest.mark.parametrize("number", [3, 4])
def test_a_tap_on_an_option_that_isnt_there_does_nothing(cfg, store, number):
    _, version = draft_with(store)
    press(store, cfg, version.id, number)
    assert store.jobs("queued") == []
    assert "no open decision with that option" in store.pending()[-1].text


def test_a_tap_on_a_settled_or_outdated_version_does_nothing(cfg, store):
    draft, version = draft_with(store)
    store.add_version(draft.id, SETTLED, "user")  # edited meanwhile
    press(store, cfg, version.id, 1)
    assert store.jobs("queued") == []
    assert "edited since" in store.pending()[-1].text

    settled, latest = draft_with(store, SETTLED, number=8)  # a second draft, already settled
    press(store, cfg, latest.id, 1)
    assert store.jobs("queued") == []


def test_choosing_then_the_revision_can_be_posted(agent_cfg, store, hist, tmp_path, monkeypatch):
    put_issue(hist)
    put_comment(hist, 5, 7, "Which one do you prefer?")
    draft, version = draft_with(store)
    press(store, agent_cfg, version.id, 2)

    fake = FakeModel()
    fake.reply = {"reply": SETTLED, "note": "Filled in option 2."}
    monkeypatch.setattr(llm, "converse", fake)
    assert worker.Worker(agent_cfg, store, hist, tmp_path).step()

    ((_, user),) = fake.prompts("reply")
    assert "option 2, store them with their zone" in user.split("The maintainer's instruction")[-1]
    latest = store.latest_version(draft.id)
    assert (latest.text, latest.author) == (SETTLED, "revised")
    offer = store.pending()[-1]
    assert [label for label, _ in offer.buttons] == ["✅ Post", "🗑 Reject"]  # now it can be posted

    store.record_decision("draft", "post", latest.id)
    fakeposter = FakePoster()
    poster.apply_decisions(store, agent_cfg, fakeposter)
    assert fakeposter.posts == [(7, SETTLED)]


def test_the_safety_net_still_refuses_an_open_decision_from_a_stale_button(cfg, store):
    _, version = draft_with(store)
    store.record_decision("draft", "post", version.id)
    fake = FakePoster()
    poster.apply_decisions(store, cfg, fake)
    assert fake.posts == [] and "still has a [YOUR DECISION" in store.pending()[-1].text


# -- the web page --------------------------------------------------------------------------------


def test_the_web_page_offers_the_choices_while_a_decision_is_open(cfg, tmp_path):
    data = web.Data(cfg, tmp_path)
    put_issue(data.history)
    draft, version = draft_with(data.store)
    found = data.draft(draft.id)

    assert found["open_decision"] is True and found["verdict"]["options"] == list(OPTIONS)
    assert [c["label"] for c in found["choices"]] == [f"1. {OPTIONS[0]}", f"2. {OPTIONS[1]}"]
    assert found["choices"][1]["instruction"] == drafts.choose_instruction(2, OPTIONS[1])

    data.store.add_version(draft.id, SETTLED, "user")
    settled = data.draft(draft.id)
    assert settled["open_decision"] is False and settled["choices"] == []
    assert render.draft  # (the Telegram text is covered above)
