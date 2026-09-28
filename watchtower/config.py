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


@dataclass(frozen=True)
class Config:
    repos: tuple[str, ...]
    poll_seconds: int
    ignore_authors: frozenset[str]
    backfill_hours: float
    chat_id: int
    allowed_user_id: int
    topics: dict[str, int | None]
    llm_url: str
    summary_model: str
    num_ctx: int
    llm_timeout: float


def is_cloud_model(name: str) -> bool:
    """Ollama's ``*-cloud`` models run on ollama.com -- prompts would leave the box."""

    return "cloud" in name.lower().rsplit(":", 1)[-1] or name.lower().endswith("-cloud")


def parse(text: str) -> Config:
    raw = tomllib.loads(text)
    github = raw.get("github", {})
    telegram = raw.get("telegram", {})
    llm = raw.get("llm", {})

    repos = tuple(github.get("repos", ()))
    if not repos or any(repo.count("/") != 1 for repo in repos):
        raise ValueError("github.repos must list at least one 'owner/name'")

    model = str(llm.get("summary_model", ""))
    if model and is_cloud_model(model):
        raise ValueError(f"llm.summary_model {model!r} is a cloud model; use a local one")

    if "chat_id" not in telegram or "allowed_user_id" not in telegram:
        raise ValueError("telegram.chat_id and telegram.allowed_user_id are required")

    topic_ids = telegram.get("topics", {})
    unknown = set(topic_ids) - set(TOPICS)
    if unknown:
        raise ValueError(f"unknown telegram.topics: {sorted(unknown)}")

    return Config(
        repos=repos,
        poll_seconds=max(60, int(github.get("poll_seconds", 120))),
        ignore_authors=frozenset(a.lower() for a in github.get("ignore_authors", ())),
        backfill_hours=float(github.get("backfill_hours", 0)),
        chat_id=int(telegram["chat_id"]),
        allowed_user_id=int(telegram["allowed_user_id"]),
        # 0 or missing means the group's General topic.
        topics={name: int(topic_ids.get(name, 0)) or None for name in TOPICS},
        llm_url=str(llm.get("url", "http://ollama:11434")).rstrip("/"),
        summary_model=model,
        num_ctx=int(llm.get("num_ctx", 16384)),
        llm_timeout=float(llm.get("timeout_seconds", 180)),
    )


def load(path: Path = CONFIG_PATH) -> Config:
    return parse(path.read_text(encoding="utf-8"))


def read_secret(name: str) -> str:
    return (SECRETS_DIR / name).read_text(encoding="utf-8").strip()
