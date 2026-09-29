"""The web process: a page on the LAN that shows how Watchtower works, what it's
doing now, and everything it did -- every message, job, draft, pass and model call.

No login: anyone on the LAN can look. So it holds no secret, and what it can do
is limited to what's harmless without one. Edits and rejections of drafts are
recorded as ``decision`` rows with origin ``web`` and applied by the poster, like
Telegram replies; "Post" only offers the version in Telegram again, and only
the allowed user's tap there posts (the poster refuses a web-origin post).

Other websites mustn't use a LAN browser to act or to read:

- a POST needs the ``X-Watchtower`` header and a JSON body, which a cross-site
  page can only send after a CORS preflight, which is never answered;
- the ``Host`` header must be an IP address, ``localhost`` or a name listed in
  ``web.hosts`` (DNS rebinding);
- a strict Content-Security-Policy; the page builds everything from outside
  (strangers' text, model output) as text, never as HTML.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import drafts, handoff, render, snapshot
from .analysis import CATEGORIES, Verdict
from .config import DATA_DIR, Config
from .history import History
from .store import Store
from .trace import Trace

_LOGGER = logging.getLogger(__name__)

STATIC = Path(__file__).with_name("web")
# The only files served, with their types.
FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
FONT = re.compile(r"/fonts/([a-z0-9-]+\.woff2)")
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:;"
        " font-src 'self'; connect-src 'self'; base-uri 'none'; form-action 'none';"
        " frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
MAX_BODY = 20_000
BEAT_SECONDS = 30
PAGE = 50


# -- what the page reads -------------------------------------------------------------


class Data:
    """The databases, read for the page; the three web actions, recorded."""

    def __init__(self, cfg: Config, data: Path = DATA_DIR) -> None:
        self.cfg = cfg
        self.data = data
        self.store = Store(data / "watchtower.db")
        self.history = History(data / "history.db")
        self.trace = Trace(data / "trace.db")

    def close(self) -> None:
        self.store.close()
        self.history.close()
        self.trace.close()

    def _rows(self, db, sql: str, args: tuple = ()) -> list[dict]:
        cursor = db.execute(sql, args)
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    # -- now ------------------------------------------------------------------

    def overview(self) -> dict:
        now = time.time()
        s = self.store.db
        count = lambda sql: dict(s.execute(sql).fetchall())  # noqa: E731
        return {
            "now": now,
            "repos": list(self.cfg.repos),
            "poll_seconds": self.cfg.poll_seconds,
            "history_minutes": self.cfg.history_minutes,
            "models": {"summary": self.cfg.summary_model, "agent": self.cfg.agent_model},
            "agent_num_ctx": self.cfg.agent_num_ctx,
            "agent_steps": self.cfg.agent_steps,
            "drafting": self.cfg.drafts,
            "heartbeats": {
                name: {"at": at, "detail": detail}
                for name, (at, detail) in self.store.heartbeats().items()
            },
            "outbox": self.store.outbox_stats(now - 86400),
            "jobs": count("SELECT status, COUNT(*) FROM job GROUP BY status"),
            "queued_jobs": count(
                "SELECT kind, COUNT(*) FROM job WHERE status IN ('queued', 'running') GROUP BY kind"
            ),
            "drafts": count("SELECT status, COUNT(*) FROM draft GROUP BY status"),
            "open_decisions": s.execute(
                "SELECT COUNT(*) FROM decision WHERE applied IS NULL"
            ).fetchone()[0],
            "last_call": next(iter(self.trace.calls(limit=1)), None),
            "in_flight": self._rows(
                s,
                "SELECT id, repo, number, title, status, updated FROM draft"
                " WHERE status IN ('prep', 'queued', 'drafting', 'posting') ORDER BY id",
            ),
        }

    # -- history --------------------------------------------------------------

    def activity(self, before: int | None = None, limit: int = PAGE) -> list[dict]:
        """What Watchtower told you, newest first: the outbox is its own record."""

        sql = (
            "SELECT id, topic, text, url, silent, created, sent, attempts, error, buttons, ref"
            " FROM outbox"
        )
        args: tuple = ()
        if before is not None:
            sql += " WHERE id < ?"
            args = (before,)
        rows = self._rows(self.store.db, f"{sql} ORDER BY id DESC LIMIT ?", (*args, limit))
        for row in rows:
            row["buttons"] = json.loads(row["buttons"] or "[]")
            row["silent"] = bool(row["silent"])
        return rows

    def jobs(self, limit: int = PAGE) -> list[dict]:
        rows = self._rows(
            self.store.db,
            "SELECT id, kind, key, status, created, updated FROM job ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return rows

    def stats(self, days: int = 30) -> dict:
        since = time.time() - days * 86400
        s = self.store.db
        per_day = lambda sql: [  # noqa: E731
            {"day": d, "count": n} for d, n in s.execute(sql, (since,)).fetchall()
        ]
        return {
            "days": days,
            "messages": per_day(
                "SELECT date(created, 'unixepoch') AS day, COUNT(*) FROM outbox"
                " WHERE created >= ? GROUP BY day ORDER BY day"
            ),
            "drafts": per_day(
                "SELECT date(created, 'unixepoch') AS day, COUNT(*) FROM draft"
                " WHERE created >= ? GROUP BY day ORDER BY day"
            ),
            "calls": self.trace.daily(since),
        }

    # -- drafts ---------------------------------------------------------------

    def drafts(self, limit: int = PAGE) -> list[dict]:
        rows = self._rows(
            self.store.db,
            "SELECT d.id, d.repo, d.number, d.kind, d.title, d.url, d.status, d.posted_url,"
            " d.verdict, d.created, d.updated,"
            " (SELECT COUNT(*) FROM draft_version v WHERE v.draft = d.id) AS versions"
            " FROM draft d ORDER BY d.id DESC LIMIT ?",
            (limit,),
        )
        for row in rows:
            verdict = Verdict.from_json(row.pop("verdict"))
            row["category"] = verdict.category if verdict else ""
            row["confidence"] = verdict.confidence if verdict else ""
        return rows

    def draft(self, draft_id: int) -> dict | None:
        found = self._rows(self.store.db, "SELECT * FROM draft WHERE id = ?", (draft_id,))
        if not found:
            return None
        draft = found[0]
        verdict = Verdict.from_json(draft.pop("verdict"))
        versions = self._rows(
            self.store.db,
            "SELECT id, text, author, created FROM draft_version WHERE draft = ? ORDER BY id",
            (draft_id,),
        )
        # Replies (edits, reasons) point at the draft, buttons at a version; a draft id
        # and a version id can be equal, so each is taken by its action.
        decisions = [d for d in self.store.decisions("draft", draft_id) if d["action"] == "reply"]
        for version in versions:
            decisions += [
                d for d in self.store.decisions("draft", version["id"]) if d["action"] != "reply"
            ]
        stages = self._rows(
            self.store.db,
            "SELECT name, data, created FROM draft_stage WHERE draft = ? ORDER BY created",
            (draft_id,),
        )
        for stage in stages:
            stage["data"] = json.loads(stage["data"])
        found = self.store.draft(draft_id)
        return {
            **draft,
            "verdict": _verdict(verdict),
            "handoff": _handoff(found, verdict),
            "versions": versions,
            "decisions": sorted(decisions, key=lambda d: d["id"]),
            "stages": stages,
            "calls": self.trace.calls(subject=f"draft:{draft_id}", limit=500)[::-1],
            "thread": self.history.thread(draft["repo"], draft["number"]),
            "max_text": drafts.MAX_SHOWN,
        }

    def calls(self, subject: str | None, before: int | None) -> list[dict]:
        return self.trace.calls(subject=subject, before=before, limit=PAGE)

    def call(self, call_id: int) -> dict | None:
        return self.trace.call(call_id)

    # -- knowledge ------------------------------------------------------------

    def knowledge(self) -> list[dict]:
        found = []
        for repo in self.cfg.repos:
            main = snapshot.path_for(self.data / "repos", repo)
            briefs = self._rows(
                self.history.db,
                "SELECT id, ref, label, text, status, created FROM brief"
                " WHERE repo = ? ORDER BY id DESC LIMIT 20",
                (repo,),
            )
            releases = self._rows(
                self.history.db,
                "SELECT tag, prerelease, published FROM release WHERE repo = ?"
                " ORDER BY published DESC LIMIT 30",
                (repo,),
            )
            found.append(
                {
                    "repo": repo,
                    "counts": self.history.counts(repo),
                    "snapshot": self.history.get_cursor(f"snapshot:{repo}"),
                    "notes": [path for path, _ in snapshot.notes(main)],
                    "docs": [path for path, _ in snapshot.trusted_docs(main)],
                    "briefs": briefs,
                    "releases": releases,
                }
            )
        return found

    # -- the web actions (never a post) ---------------------------------------

    def edit(self, draft_id: int, text: str) -> str:
        draft = self.store.draft(draft_id)
        text = text.strip()
        if draft is None:
            raise ValueError("no such draft")
        if not text:
            raise ValueError("the text is empty")
        if len(text) > drafts.MAX_SHOWN:
            raise ValueError(f"at most {drafts.MAX_SHOWN} characters")
        if draft.status != "ready":
            raise ValueError(f"the draft is {draft.status}")
        self.store.record_decision("draft", "reply", draft_id, text, origin="web")
        return "Sent: it comes back in Telegram as a new version, with its own buttons."

    def revise(self, draft_id: int, instruction: str) -> str:
        """An instruction for the model: the worker revises the latest version."""

        draft = self.store.draft(draft_id)
        instruction = instruction.strip()
        if draft is None:
            raise ValueError("no such draft")
        if not instruction:
            raise ValueError("say what to change")
        if len(instruction) > drafts.MAX_INSTRUCTION:
            raise ValueError(f"at most {drafts.MAX_INSTRUCTION} characters")
        if draft.status != "ready":
            raise ValueError(f"the draft is {draft.status}")
        self.store.record_decision("draft", "revise", draft_id, instruction, origin="web")
        return "Sent: the model revises it, and it comes back here and in Telegram with its own buttons."

    def _version(self, version_id: int):
        version = self.store.version(version_id)
        if version is None:
            raise ValueError("no such version")
        return version

    def reject(self, version_id: int) -> str:
        self._version(version_id)
        self.store.record_decision("draft", "reject", version_id, origin="web")
        return "Rejected. Telegram will say so; reply to the draft there with a reason if you like."

    def offer(self, version_id: int) -> str:
        self._version(version_id)
        self.store.record_decision("draft", "offer", version_id, origin="web")
        return "Sent to Telegram: tap ✅ Post there to post it."


def _handoff(draft, verdict: Verdict | None) -> dict | None:
    """A bug's prompt for Claude Code (``handoff``), and whether it's confirmed."""

    if draft is None or not handoff.is_bug(verdict):
        return None
    return {
        "confirmed": handoff.confirmed(verdict),
        "doubts": handoff.doubts(verdict),
        "label": handoff.label(draft),
        "prompt": handoff.prompt(draft, verdict),
    }


def _verdict(verdict: Verdict | None) -> dict | None:
    if verdict is None:
        return None
    return {
        "category": verdict.category,
        "label": CATEGORIES.get(verdict.category, verdict.category),
        "confidence": verdict.confidence,
        "evidence": [vars(e) for e in verdict.evidence],
        "missing": list(verdict.missing),
        "code": verdict.code,
        "fix": verdict.fix,
        "asks": [vars(a) for a in verdict.asks],
        "decision": verdict.decision,
        "attempts": verdict.attempts,
        "looked_at": list(verdict.looked_at),
        "unknown_paths": list(verdict.unknown_paths),
        "judged_at": verdict.judged_at,
    }


# -- HTTP -------------------------------------------------------------------------------


def allowed_host(host: str, names: frozenset[str]) -> bool:
    """An IP address, ``localhost``, or a name the config lists: a page from another
    site that rebinds its own name to this box's address sends that name."""

    host = host.strip().lower()
    if host.startswith("["):  # [::1]:8080
        host = host[1 : host.find("]")]
    elif host.count(":") == 1:
        host = host.split(":")[0]
    if not host:
        return False
    if host == "localhost" or host in names:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


GET_ROUTES = [
    (re.compile(r"/api/overview"), "overview"),
    (re.compile(r"/api/activity"), "activity"),
    (re.compile(r"/api/jobs"), "jobs"),
    (re.compile(r"/api/stats"), "stats"),
    (re.compile(r"/api/drafts"), "drafts"),
    (re.compile(r"/api/drafts/(\d+)"), "draft"),
    (re.compile(r"/api/calls"), "calls"),
    (re.compile(r"/api/calls/(\d+)"), "call"),
    (re.compile(r"/api/knowledge"), "knowledge"),
]
POST_ROUTES = [
    (re.compile(r"/api/drafts/(\d+)/edit"), "edit"),
    (re.compile(r"/api/drafts/(\d+)/revise"), "revise"),
    (re.compile(r"/api/versions/(\d+)/reject"), "reject"),
    (re.compile(r"/api/versions/(\d+)/offer"), "offer"),
]


def _int(query: dict, name: str) -> int | None:
    value = (query.get(name) or [""])[0]
    return int(value) if value.isdigit() else None


def answer_get(data: Data, path: str, query: dict) -> tuple[int, Any]:
    for pattern, name in GET_ROUTES:
        match = pattern.fullmatch(path)
        if match is None:
            continue
        arg = int(match[1]) if match.groups() else None
        match name:
            case "activity":
                return 200, data.activity(_int(query, "before"))
            case "stats":
                return 200, data.stats(min(_int(query, "days") or 30, 365))
            case "calls":
                subject = (query.get("subject") or [None])[0]
                return 200, data.calls(subject, _int(query, "before"))
            case "draft" | "call":
                found = getattr(data, name)(arg)
                return (200, found) if found is not None else (404, {"error": "not found"})
            case _:
                return 200, getattr(data, name)()
    return 404, {"error": "not found"}


def answer_post(data: Data, path: str, body: dict) -> tuple[int, Any]:
    for pattern, name in POST_ROUTES:
        match = pattern.fullmatch(path)
        if match is None:
            continue
        try:
            if name in ("edit", "revise"):
                text = body.get("text")
                message = getattr(data, name)(int(match[1]), text if isinstance(text, str) else "")
            else:
                message = getattr(data, name)(int(match[1]))
        except ValueError as err:
            return 400, {"error": str(err)}
        return 200, {"message": message}
    return 404, {"error": "not found"}


class Handler(BaseHTTPRequestHandler):
    cfg: Config
    data_dir: Path = DATA_DIR
    server_version = "watchtower"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        _LOGGER.debug("%s %s", self.address_string(), format % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        for name, value in HEADERS.items():
            self.send_header(name, value)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        body = json.dumps(value, ensure_ascii=False).encode()
        self._send(status, body, "application/json; charset=utf-8")

    def _guard(self) -> bool:
        if not allowed_host(self.headers.get("Host", ""), frozenset(self.cfg.web_hosts)):
            self._json(421, {"error": "unknown host name: add it to web.hosts in config.toml"})
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802
        if not self._guard():
            return
        url = urlsplit(self.path)
        if url.path in FILES:
            name, content_type = FILES[url.path]
            self._send(200, (STATIC / name).read_bytes(), content_type)
            return
        font = FONT.fullmatch(url.path)
        if font is not None and (STATIC / "fonts" / font[1]).is_file():
            self._send(200, (STATIC / "fonts" / font[1]).read_bytes(), "font/woff2")
            return
        if not url.path.startswith("/api/"):
            self._json(404, {"error": "not found"})
            return
        data = Data(self.cfg, self.data_dir)
        try:
            status, value = answer_get(data, url.path, parse_qs(url.query))
        finally:
            data.close()
        self._json(status, value)

    def do_POST(self) -> None:  # noqa: N802
        if not self._guard():
            return
        # A cross-site page can't send this header or JSON without a preflight.
        if self.headers.get("X-Watchtower") != "1" or not self.headers.get(
            "Content-Type", ""
        ).startswith("application/json"):
            self._json(403, {"error": "missing X-Watchtower header or JSON body"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._json(413, {"error": "too big"})
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self._json(400, {"error": "a JSON object, please"})
            return
        data = Data(self.cfg, self.data_dir)
        try:
            status, value = answer_post(data, urlsplit(self.path).path, body)
        finally:
            data.close()
        self._json(status, value)

    def do_OPTIONS(self) -> None:  # noqa: N802
        # No CORS: a preflight from another site gets no allowance.
        self._json(405, {"error": "no cross-site requests"})


def _beat(path: Path) -> None:
    store = Store(path)
    while True:
        store.beat("web", "")
        time.sleep(BEAT_SECONDS)


def run(cfg: Config) -> None:
    Store(DATA_DIR / "watchtower.db").enqueue(
        "system", render.system(f"Web UI online on port {cfg.web_port}.")
    )
    threading.Thread(
        target=_beat, args=(DATA_DIR / "watchtower.db",), daemon=True, name="beat"
    ).start()
    handler = type("ConfiguredHandler", (Handler,), {"cfg": cfg})
    server = ThreadingHTTPServer(("0.0.0.0", cfg.web_port), handler)  # noqa: S104 -- the LAN
    _LOGGER.info("web UI on port %d", cfg.web_port)
    server.serve_forever()
