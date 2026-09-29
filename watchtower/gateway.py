"""The gateway process: the only holder of the Telegram token and the only Telegram poller.

Telegram lets exactly one process call ``getUpdates`` per bot, so every agent
talks to Telegram through here: agents write to the outbox, the gateway sends.
Incoming messages and button presses count only when they come from the
configured user in the configured group; everything else is dropped without a
reply. A press, or a reply to a message that takes replies (a draft), is only
recorded here (``decision``); the watcher (briefs) or the poster (drafts) applies it.
"""

from __future__ import annotations

import logging
import threading
import time

from . import render
from .config import DATA_DIR, Config, read_secret
from .store import Store
from .telegram import Bot, TelegramError, to_markdown

_LOGGER = logging.getLogger(__name__)

# Telegram allows about 20 messages a minute into one group.
SEND_INTERVAL = 3.2
# What each kind of button may do, with the confirmation shown on a press.
BUTTON_ACTIONS = {
    "brief": {"approve": "Brief approved", "reject": "Brief discarded"},
    "draft": {
        "post": "Posting…",
        "plain": "Posting, without the label…",
        "reject": "Rejected. Reply to the draft with a reason if you like.",
    },
}
# Messages whose replies count (their outbox ``ref`` is ``kind:id``).
REPLY_KINDS = frozenset({"draft"})


def authorized(update: dict, cfg: Config) -> dict | None:
    """The update's message if it's from the allowed user in the configured group."""

    message = update.get("message") or {}
    if (message.get("chat") or {}).get("id") != cfg.chat_id:
        return None
    if (message.get("from") or {}).get("id") != cfg.allowed_user_id:
        return None
    return message


def authorized_press(update: dict, cfg: Config) -> dict | None:
    """The update's button press if it's the allowed user's, on a message in the group."""

    query = update.get("callback_query") or {}
    if (query.get("from") or {}).get("id") != cfg.allowed_user_id:
        return None
    if ((query.get("message") or {}).get("chat") or {}).get("id") != cfg.chat_id:
        return None
    return query


def parse_press(data: str | None) -> tuple[str, str, int] | None:
    """``kind:action:ref`` as sent by our own buttons, or ``None`` for anything else."""

    parts = (data or "").split(":")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    kind, action, ref = parts
    if action not in BUTTON_ACTIONS.get(kind, ()):
        return None
    return kind, action, int(ref)


def press(bot: Bot, store: Store, cfg: Config, query: dict) -> None:
    """Record the decision first (it counts even if Telegram fails next), then confirm
    and remove the buttons so it can't be pressed twice."""

    parsed = parse_press(query.get("data"))
    answer = {"callback_query_id": query["id"]}
    if parsed is not None:
        store.record_decision(*parsed)
        answer["text"] = BUTTON_ACTIONS[parsed[0]][parsed[1]]
    bot.call("answerCallbackQuery", answer)
    if parsed is not None:
        bot.call(
            "editMessageReplyMarkup",
            {
                "chat_id": cfg.chat_id,
                "message_id": query["message"]["message_id"],
                "reply_markup": {"inline_keyboard": []},
            },
        )


def reply_target(store: Store, message: dict) -> tuple[str, int] | None:
    """``(kind, id)`` if ``message`` replies to one of ours that takes replies."""

    replied = (message.get("reply_to_message") or {}).get("message_id")
    if not isinstance(replied, int):
        return None
    kind, _, ref = (store.ref_for_message(replied) or "").partition(":")
    if kind not in REPLY_KINDS or not ref.isdigit():
        return None
    return kind, int(ref)


def note_reply(store: Store, message: dict) -> bool:
    """Record a reply to a draft (an edit, or a reason). Returns whether it was one."""

    target = reply_target(store, message)
    if target is None:
        return False
    kind, ref = target
    text = to_markdown(message.get("text") or "", message.get("entities"))
    if text.strip():
        store.record_decision(kind, "reply", ref, text)
    return True


def command(message: dict) -> str | None:
    text = (message.get("text") or "").strip()
    if not text.startswith("/"):
        return None
    return text.split()[0].split("@", 1)[0].lower()


def _ago(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def status_text(store: Store, now: float) -> str:
    lines = ["Status"]
    for name, (at, detail) in sorted(store.heartbeats().items()):
        lines.append(f"• {name}: {_ago(now - at)}" + (f" ({detail})" if detail else ""))
    stats = store.outbox_stats(now - 86400)
    lines.append(
        f"• outbox: {stats['pending']} pending, {stats['failed']} failed,"
        f" {stats['sent']} sent in 24 h"
    )
    return render.system("\n".join(lines))


def reply(bot: Bot, cfg: Config, message: dict, text: str) -> None:
    bot.send(cfg.chat_id, text, thread_id=message.get("message_thread_id"))


def deliver(bot: Bot, store: Store, cfg: Config) -> bool:
    """Send what's pending. Returns whether anything was attempted."""

    batch = store.pending()
    for item in batch:
        try:
            sent = bot.send(
                cfg.chat_id,
                item.text,
                thread_id=cfg.topics.get(item.topic),
                url=item.url,
                silent=item.silent,
                buttons=item.buttons,
            )
        except TelegramError as err:
            if err.retry_after:
                _LOGGER.warning("Telegram asks to wait %ss", err.retry_after)
                time.sleep(float(err.retry_after) + 1)
                return True
            _LOGGER.warning("send %s failed: %s", item.id, err)
            store.mark_failed(item.id, str(err), permanent=err.permanent)
        else:
            store.mark_sent(item.id, (sent or {}).get("message_id"))
        time.sleep(SEND_INTERVAL)
    return bool(batch)


def _sender(bot: Bot, cfg: Config) -> None:
    store = Store(DATA_DIR / "watchtower.db")
    while True:
        try:
            if not deliver(bot, store, cfg):
                time.sleep(2)
        except Exception:
            _LOGGER.exception("sender loop error")
            time.sleep(10)


def run(cfg: Config) -> None:
    bot = Bot(read_secret("telegram_token"))
    store = Store(DATA_DIR / "watchtower.db")
    me = bot.call("getMe")
    _LOGGER.info("gateway started as @%s", me.get("username"))
    store.enqueue("system", render.system(f"Gateway online as @{me.get('username')}."))
    threading.Thread(target=_sender, args=(bot, cfg), daemon=True, name="sender").start()

    offset = int(store.get_cursor("telegram:offset") or 0)
    while True:
        store.beat("gateway", "")
        try:
            updates = bot.call(
                "getUpdates",
                {"offset": offset, "timeout": 25, "allowed_updates": ["message", "callback_query"]},
                timeout=40,
            )
        except TelegramError as err:
            _LOGGER.warning("getUpdates failed: %s", err)
            time.sleep(err.retry_after or 5)
            continue
        for update in updates:
            offset = update["update_id"] + 1
            store.set_cursor("telegram:offset", str(offset))
            if (query := authorized_press(update, cfg)) is not None:
                try:
                    press(bot, store, cfg, query)
                except TelegramError as err:
                    _LOGGER.warning("button press: %s", err)
                continue
            message = authorized(update, cfg)
            if message is None or (command(message) is None and note_reply(store, message)):
                continue
            match command(message):
                case "/status":
                    reply(bot, cfg, message, status_text(store, time.time()))
                case "/ping":
                    reply(bot, cfg, message, render.system("pong"))
                case _:
                    pass
