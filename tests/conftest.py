"""Make ``watchtower`` importable and provide a config and a fresh store.

Run from the repo root: ``python -m pytest ops/watchtower/tests``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from watchtower import config  # noqa: E402
from watchtower.store import Store  # noqa: E402

CONFIG = """
[github]
repos = ["owner/repo"]
ignore_authors = ["Owner"]

[telegram]
chat_id = -1001
allowed_user_id = 42

[telegram.topics]
triage = 11
system = 0

[llm]
summary_model = "tiny:1b"
"""


@pytest.fixture
def cfg() -> config.Config:
    return config.parse(CONFIG)


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "w.db")
    yield s
    s.close()
