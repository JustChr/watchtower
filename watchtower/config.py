"""Configuration: one TOML file for settings, one file per secret."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(os.environ.get("WATCHTOWER_CONFIG", "/config/config.toml"))
SECRETS_DIR = Path(os.environ.get("WATCHTOWER_SECRETS", "/run/secrets"))
DATA_DIR = Path(os.environ.get("WATCHTOWER_DATA", "/data"))

TOPICS = ("triage", "reviews", "replies", "system")
THINK_LEVELS = frozenset({"", "low", "medium", "high"})  # Ollama's reasoning efforts


@dataclass(frozen=True)
class Config:
    repos: tuple[str, ...]
    poll_seconds: int
    ignore_authors: frozenset[str]
    backfill_hours: float
    history_minutes: int
    chat_id: int
    allowed_user_id: int
    topics: dict[str, int | None]
    llm_url: str
    summary_model: str
    num_ctx: int
    llm_timeout: float
    agent_model: str
    agent_num_ctx: int
    agent_timeout: float
    agent_think: str  # reasoning effort for agent work: low/medium/high; "" = the model's default
    agent_steps: int  # rounds of tool calls before an assessment; 0 = no tools
    brief_betas: bool
    draft_replies: bool
    app_id: str  # the GitHub App that posts approved replies; "" = posting off
    review_prs: bool = True  # draft a review of every stranger's new pull request
    run_checks: bool = False  # run the PR's checks in the sandbox (the runner service)
    web_port: int = 8080
    web_hosts: tuple[str, ...] = ()  # host names the web UI answers to, besides IPs

    @property
    def drafts(self) -> bool:
        """Draft replies are on: asked for, and there's a model to write them."""

        return self.draft_replies and bool(self.agent_model)

    @property
    def reviews(self) -> bool:
        """PR reviews are on: drafts are, and the config asks for them."""

        return self.drafts and self.review_prs


def is_cloud_model(name: str) -> bool:
    """Ollama's ``*-cloud`` models run on ollama.com -- prompts would leave the box."""

    return "cloud" in name.lower().rsplit(":", 1)[-1] or name.lower().endswith("-cloud")


def parse(text: str) -> Config:
    raw = tomllib.loads(text)
    github = raw.get("github", {})
    telegram = raw.get("telegram", {})
    llm = raw.get("llm", {})
    web = raw.get("web", {})

    repos = tuple(github.get("repos", ()))
    if not repos or any(repo.count("/") != 1 for repo in repos):
        raise ValueError("github.repos must list at least one 'owner/name'")

    model = str(llm.get("summary_model", ""))
    agent_model = str(llm.get("agent_model", ""))
    for key, name in (("summary_model", model), ("agent_model", agent_model)):
        if name and is_cloud_model(name):
            raise ValueError(f"llm.{key} {name!r} is a cloud model; use a local one")

    agent_think = str(llm.get("agent_think", "high")).strip().lower()
    if agent_think not in THINK_LEVELS:
        raise ValueError(f"llm.agent_think must be one of {sorted(THINK_LEVELS)}")

    if "chat_id" not in telegram or "allowed_user_id" not in telegram:
        raise ValueError("telegram.chat_id and telegram.allowed_user_id are required")

    topic_ids = telegram.get("topics", {})
    unknown = set(topic_ids) - set(TOPICS)
    if unknown:
        raise ValueError(f"unknown telegram.topics: {sorted(unknown)}")

    app_id = str(github.get("app_id", "")).strip()
    if app_id and not app_id.isalnum():
        raise ValueError("github.app_id must be the App's ID or Client ID")

    # 0 turns the history off; otherwise it syncs at most every 10 minutes.
    history_minutes = int(github.get("history_minutes", 60))
    if history_minutes:
        history_minutes = max(10, history_minutes)

    return Config(
        repos=repos,
        poll_seconds=max(60, int(github.get("poll_seconds", 120))),
        ignore_authors=frozenset(a.lower() for a in github.get("ignore_authors", ())),
        backfill_hours=float(github.get("backfill_hours", 0)),
        history_minutes=history_minutes,
        chat_id=int(telegram["chat_id"]),
        allowed_user_id=int(telegram["allowed_user_id"]),
        # 0 or missing means the group's General topic.
        topics={name: int(topic_ids.get(name, 0)) or None for name in TOPICS},
        llm_url=str(llm.get("url", "http://ollama:11434")).rstrip("/"),
        summary_model=model,
        num_ctx=int(llm.get("num_ctx", 16384)),
        llm_timeout=float(llm.get("timeout_seconds", 180)),
        agent_model=agent_model,
        agent_num_ctx=int(llm.get("agent_num_ctx", 32768)),
        agent_timeout=float(llm.get("agent_timeout_seconds", 900)),
        agent_think=agent_think,
        agent_steps=max(0, int(llm.get("agent_steps", 20))),
        brief_betas=bool(llm.get("brief_betas", True)),
        draft_replies=bool(llm.get("draft_replies", True)),
        app_id=app_id,
        review_prs=bool(llm.get("review_prs", True)),
        run_checks=bool(llm.get("run_checks", False)),
        web_port=int(web.get("port", 8080)),
        web_hosts=tuple(str(h).strip().lower() for h in web.get("hosts", ())),
    )


def load(path: Path = CONFIG_PATH) -> Config:
    return parse(path.read_text(encoding="utf-8"))


def read_secret(name: str) -> str:
    return (SECRETS_DIR / name).read_text(encoding="utf-8").strip()
