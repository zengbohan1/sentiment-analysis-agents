"""pytest 共享夹具：MockLLM、内存缓存、ForumEngine、TaskManager。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# 保证 app 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LLM_MOCK", "true")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")

from app.config import Settings, get_settings  # noqa: E402
from app.core.cache import Cache  # noqa: E402
from app.core.llm import MockLLM  # noqa: E402
from app.core.observability import Observability  # noqa: E402
from app.core.task_manager import TaskManager  # noqa: E402
from app.engines.forum_engine import ForumEngine  # noqa: E402
from app.engines.report_engine import ReportEngine  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def cache() -> Cache:
    c = Cache(get_settings())
    return c


@pytest.fixture
def observability() -> Observability:
    return Observability(get_settings())


@pytest.fixture
def forum(mock_llm, observability) -> ForumEngine:
    return ForumEngine(mock_llm, observability=observability)


@pytest.fixture
def report_engine() -> ReportEngine:
    return ReportEngine(get_settings())


@pytest.fixture
def task_manager(mock_llm, cache, observability) -> TaskManager:
    return TaskManager(get_settings(), mock_llm, cache, observability)
