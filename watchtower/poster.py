"""The poster process: the only holder of the GitHub App's key, so the only thing
that can write to GitHub. It never talks to the model.

It applies the user's decisions on drafts, as the gateway recorded them:

- ✅ Post on a version posts exactly that text, if it's still the draft's
  latest version and the draft is still ready (not rejected, not superseded),
  and no ``[YOUR DECISION: ...]`` line is left in it; for a confirmed bug in
  an issue it also adds the label the message named (``handoff.label``), which
  "✅ Post only" leaves out;
- 🗑 Reject rejects the draft;
- a Telegram reply to a ready draft is an instruction: the worker has the model
  revise the latest version as it says (a ``revise`` job), and offers the result
  with its own buttons; a reply starting with ``text:`` is the user's own text,
  used as it is. A reply to a rejected draft is the reason.

The web UI records the same instructions, edits (its editor: always the text
as it is) and rejections (origin ``web``). It can't
post: its "Post" only offers that version in Telegram again (``offer``), and a
post decision from anywhere but Telegram is refused.

A post is claimed (``posting``) before GitHub is called, so it never happens
twice. If it may have gone through after all (timeout, 5xx, restart), the user
is told to check GitHub instead of it being retried.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from . import drafts, handoff, render
from .config import DATA_DIR, Config, read_secret
from .github import GitHubError
from .store import Draft, Store, Version

_LOGGER = logging.getLogger(__name__)

MAX_REASON = 2000
WEB_OFFER = "🌐 Sent from the web UI: ✅ Post posts exactly this version."


class Poster(Protocol):
    def post(self, draft: Draft, body: str) -> str: ...
    def label(self, draft: Draft, name: str) -> None: ...


class NoApp:
    """Stands in while ``github.app_id`` is unset: every post fails, visibly."""

    def post(self, draft: Draft, body: str) -> str:
        raise GitHubError("no GitHub App configured (github.app_id)", definite=True)

    def label(self, draft: Draft, name: str) -> None:
        raise GitHubError("no GitHub App configured (github.app_id)", definite=True)


def _current(store: Store, version_id: int) -> tuple[Draft, Version] | None:
    """The version's draft, if the draft is ready and this is its latest version."""

    version = store.version(version_id)
    draft = store.draft(version.draft) if version else None
    if draft is None or draft.status != "ready":
        return None
    if store.latest_version(draft.id).id != version.id:
        return None
    return draft, version


def _stale(store: Store, version_id: int) -> None:
    version = store.version(version_id)
    draft = store.draft(version.draft) if version else None
    if draft is None:
        return
    if draft.status == "ready":
        text = f"#{draft.number}: that version was edited since; use the newest draft message."
    else:
        text = f"#{draft.number}: the draft is already {draft.status}; nothing done."
    store.enqueue(draft.topic, render.system(text))


def post(
    store: Store,
    cfg: Config,
    poster: Poster,
    decision_id: int,
    version_id: int,
    labelled: bool = True,
) -> None:
    """Post the version; ``labelled``: with the label its message offered, if any."""

    current = _current(store, version_id)
    if current is None or current[0].repo not in cfg.repos:
        store.mark_applied(decision_id)
        _stale(store, version_id)
        return
    draft, version = current
    if store.decision_origin(decision_id) != "telegram":
        store.mark_applied(decision_id)
        _LOGGER.warning("post decision %d not from Telegram: refused", decision_id)
        return
    if drafts.open_decision(version.text):
        store.mark_applied(decision_id)
        text = (
            f"#{draft.number}: the draft still has a [YOUR DECISION: …] line."
            " Reply to it with your decision and the model fills it in, or with your own"
            f" version starting with {drafts.OWN_TEXT}"
        )
        store.enqueue(draft.topic, render.system(text), url=draft.url)
        return
    if not store.begin_post(draft.id, decision_id):
        return
    try:
        url = poster.post(draft, version.text)
    except Exception as err:  # noqa: BLE001 -- reported to the user either way
        definite = isinstance(err, GitHubError) and err.definite
        _LOGGER.warning("posting draft %d failed (definite: %s): %s", draft.id, definite, err)
        if definite:
            store.post_failed(draft.id)
            drafts.offer(store, draft, version, error=f"Posting failed: {err}")
        else:
            store.fail_draft(draft.id)
            text = (
                f"Posting the reply to #{draft.number} may or may not have worked"
                f" ({type(err).__name__}). Check GitHub; it won't be retried."
            )
            store.enqueue(draft.topic, render.system(text), url=draft.url)
        return
    store.posted(draft.id, url)
    tag = handoff.label(draft) if labelled else ""
    if tag:
        try:
            poster.label(draft, tag)
        except Exception as err:  # noqa: BLE001 -- the reply is out; say what's missing
            _LOGGER.warning("labelling draft %d failed: %s", draft.id, err)
            text = f"#{draft.number}: the reply is posted, but adding the label {tag} failed: {err}"
            store.enqueue(draft.topic, render.system(text), url=draft.url)
            tag = ""
    store.enqueue(draft.topic, render.posted(draft, tag), url=url)
    _LOGGER.info("posted draft %d: %s", draft.id, url)


def reject(store: Store, decision_id: int, version_id: int) -> None:
    current = _current(store, version_id)
    if current is None:
        _stale(store, version_id)
    else:
        store.reject_draft(current[0].id)
    store.mark_applied(decision_id)


def web_offer(store: Store, decision_id: int, version_id: int) -> None:
    """The web UI's "Post": show the version in Telegram again, for your tap."""

    current = _current(store, version_id)
    if current is None:
        _stale(store, version_id)
    else:
        drafts.offer(store, *current, lead=WEB_OFFER)
    store.mark_applied(decision_id)


def reply(store: Store, decision_id: int, draft_id: int, text: str, literal: bool = False) -> None:
    """A reply to a draft: an instruction for the model, unless ``literal`` or it
    starts with ``drafts.OWN_TEXT`` (then it's the new version as it is)."""

    draft = store.draft(draft_id)
    text = text.strip()
    own = text if literal else drafts.own_text(text)
    if draft is None or not text:
        pass
    elif draft.status == "ready" and own is None:
        revise(store, draft, text)
    elif draft.status == "ready":
        text = own
        if not text:
            store.enqueue(draft.topic, render.system(f"#{draft.number}: no text after text:"))
        elif len(text) > drafts.MAX_SHOWN:
            note = (
                f"#{draft.number}: your version has {len(text)} characters; drafts can show"
                f" {drafts.MAX_SHOWN} in full. Send a shorter one."
            )
            store.enqueue(draft.topic, render.system(note))
        else:
            drafts.offer(store, draft, store.add_version(draft.id, text, "user"))
    elif draft.status == "rejected":
        store.set_reason(draft.id, text[:MAX_REASON])
        store.enqueue(draft.topic, render.system(f"#{draft.number}: reason noted."))
    else:
        text = f"#{draft.number}: the draft is {draft.status}; your reply wasn't used."
        store.enqueue(draft.topic, render.system(text))
    store.mark_applied(decision_id)


def revise(store: Store, draft: Draft, instruction: str) -> None:
    """Queue the model's revision of the draft's latest version."""

    if len(instruction) > drafts.MAX_INSTRUCTION:
        text = (
            f"#{draft.number}: instructions can have {drafts.MAX_INSTRUCTION} characters;"
            f" to send your own version, start the reply with {drafts.OWN_TEXT}"
        )
        store.enqueue(draft.topic, render.system(text))
        return
    store.add_job(
        "revise",
        f"revise:{draft.id}:{time.time()}",
        {"draft": draft.id, "instruction": instruction},
    )
    store.enqueue(draft.topic, render.system(f"#{draft.number}: revising it as you asked…"))


def choose(store: Store, decision_id: int, version_id: int, number: int) -> None:
    """A tap on option ``number`` of the open decision: the model settles it that way
    (a ``revise`` job), and the result comes back with its own buttons. The option's text
    is taken from the assessment, not from the button."""

    current = _current(store, version_id)
    if current is None:
        _stale(store, version_id)
    else:
        draft, version = current
        options = drafts.options_of(draft)
        if drafts.open_decision(version.text) and 1 <= number <= len(options):
            revise(store, draft, drafts.choose_instruction(number, options[number - 1]))
        else:
            text = f"#{draft.number}: there is no open decision with that option."
            store.enqueue(draft.topic, render.system(text))
    store.mark_applied(decision_id)


def apply_decisions(store: Store, cfg: Config, poster: Poster) -> None:
    for decision_id, action, ref, text in store.open_decisions("draft"):
        match action:
            case "post":
                post(store, cfg, poster, decision_id, ref)
            case "plain":
                post(store, cfg, poster, decision_id, ref, labelled=False)
            case "reject":
                reject(store, decision_id, ref)
            case "reply":
                web = store.decision_origin(decision_id) == "web"  # the web's editor
                reply(store, decision_id, ref, text or "", literal=web)
            case "revise":
                reply(store, decision_id, ref, text or "")
            case "offer":
                web_offer(store, decision_id, ref)
            case "opt1" | "opt2" | "opt3" | "opt4":
                choose(store, decision_id, ref, int(action[3:]))
            case _:
                store.mark_applied(decision_id)


def _connect(cfg: Config) -> tuple[Poster, str]:
    if not cfg.app_id:
        return NoApp(), "no github.app_id, so nothing can be posted"
    from .github_app import App

    app = App(cfg.app_id, read_secret("github_app_key"))
    try:
        return app, f"posting as {app.slug()}[bot]"
    except GitHubError as err:
        return app, f"App check failed ({err}); posting will be tried anyway"


def run(cfg: Config) -> None:
    store = Store(DATA_DIR / "watchtower.db")
    poster, status = _connect(cfg)
    _LOGGER.info("poster started: %s", status)
    for draft in store.interrupted_posts():
        text = (
            f"A restart interrupted posting the reply to #{draft.number}."
            " Check GitHub whether it went through; it won't be retried."
        )
        store.enqueue(draft.topic, render.system(text), url=draft.url)
    store.enqueue("system", render.system(f"Poster online: {status}."))
    while True:
        store.beat("poster", "")
        try:
            apply_decisions(store, cfg, poster)
        except Exception:  # noqa: BLE001
            _LOGGER.exception("poster loop error")
            time.sleep(10)
        time.sleep(2)
