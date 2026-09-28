"""Minimal Telegram Bot API client. The token is part of every URL, so no URL is ever logged."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


# Telegram formatting entity → Markdown around it. Clients turn typed **bold** or
# `code` into entities and drop the marks, so a reply is put back together here.
_MARKS = {
    "bold": ("**", "**"),
    "italic": ("_", "_"),
    "strikethrough": ("~~", "~~"),
    "code": ("`", "`"),
}


def to_markdown(text: str, entities: list[dict] | None) -> str:
    """A message's text with its formatting as GitHub Markdown.

    Entity offsets count UTF-16 code units, so the text is cut in that encoding.
    """

    inserts: list[tuple[int, int, int, str]] = []
    for entity in entities or ():
        start = entity["offset"]
        end = start + entity["length"]
        kind = entity.get("type")
        if kind in _MARKS:
            opening, closing = _MARKS[kind]
        elif kind == "pre":
            opening, closing = f"```{entity.get('language', '')}\n", "\n```"
        elif kind == "text_link":
            opening, closing = "[", f"]({entity.get('url', '')})"
        else:
            continue
        # At one position: closings before openings; inner closes first, outer opens first.
        inserts.append((start, 1, -end, opening))
        inserts.append((end, 0, -start, closing))
    units = text.encode("utf-16-le")
    pieces, done = [], 0
    for position, _, _, mark in sorted(inserts):
        pieces += [units[done * 2 : position * 2].decode("utf-16-le"), mark]
        done = position
    pieces.append(units[done * 2 :].decode("utf-16-le"))
    return "".join(pieces)


class TelegramError(Exception):
    def __init__(self, code: int, description: str, retry_after: float | None = None) -> None:
        super().__init__(f"{code}: {description}")
        self.code = code
        self.description = description
        self.retry_after = retry_after

    @property
    def permanent(self) -> bool:
        """A 4xx other than 429 won't succeed on retry (bad chat, deleted topic, bad markup)."""

        return 400 <= self.code < 500 and self.code != 429


class Bot:
    def __init__(self, token: str) -> None:
        self._base = f"https://api.telegram.org/bot{token}/"

    def call(
        self, method: str, params: dict[str, Any] | None = None, *, timeout: float = 30
    ) -> Any:
        request = urllib.request.Request(
            self._base + method,
            data=json.dumps(params or {}).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as err:
            try:
                body = json.load(err)
            except ValueError:
                body = {"description": str(err.reason)}
            retry = (body.get("parameters") or {}).get("retry_after")
            raise TelegramError(err.code, body.get("description", "?"), retry) from None
        except (urllib.error.URLError, OSError, ValueError) as err:
            raise TelegramError(0, type(err).__name__) from None
        if not body.get("ok"):
            raise TelegramError(body.get("error_code", 0), body.get("description", "?"))
        return body["result"]

    def send(
        self,
        chat_id: int,
        text: str,
        *,
        thread_id: int | None = None,
        url: str | None = None,
        silent: bool = False,
        buttons: tuple[tuple[str, str], ...] = (),
    ) -> Any:
        params: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "link_preview_options": {"is_disabled": True},
            "disable_notification": silent,
        }
        if thread_id:
            params["message_thread_id"] = thread_id
        rows = []
        if buttons:
            rows.append([{"text": label, "callback_data": data} for label, data in buttons])
        if url and url.startswith("https://github.com/"):
            rows.append([{"text": "Open on GitHub", "url": url}])
        if rows:
            params["reply_markup"] = {"inline_keyboard": rows}
        return self.call("sendMessage", params)
