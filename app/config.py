"""全局配置（pydantic-settings，支持 .env 覆盖）。"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ---- 服务 ----
    app_name: str = "sentiment-analysis-agents"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    # ---- LLM（OpenAI 兼容协议，DeepSeek / 任意兼容网关）----
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 4096
    # 无 Key 时使用 MockLLM（确定性回复），保证离线可跑
    llm_mock: bool = False

    # ---- 数据库（SQLAlchemy 异步双数据源）----
    # PostgreSQL 主库（生产） / SQLite 兜底（测试/离线）
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/sentiment"
    # MySQL 从库（可空，空则复用主库）
    database_url_mysql: str = ""

    # ---- Redis（热点缓存 / 任务进度，宕机自动降级内存）----
    redis_url: str = "redis://localhost:6379/1"

    # ---- 任务管控 ----
    task_queue_size: int = 200
    task_concurrency: int = 8
    task_timeout_seconds: int = 1800

    # ---- 可观测 ----
    langsmith_api_key: str = ""
    langsmith_project: str = "sentiment-analysis-agents"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    # 单 token 成本（元），用于成本统计；默认按 DeepSeek 参考价估算
    token_cost_input_per_1k: float = 0.001
    token_cost_output_per_1k: float = 0.002

    # ---- 评测 ----
    eval_set_path: str = "data/eval/eval_set.jsonl"

    @property
    def use_mock_llm(self) -> bool:
        return self.llm_mock or not self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
