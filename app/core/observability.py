"""可观测：token 成本统计 + LangSmith 兼容链路追踪（可选）。"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from .llm import TokenUsage


@dataclass
class Span:
    """一次 Agent/工具调用的追踪跨度。"""

    name: str
    task_id: str
    run_id: str = ""
    parent_run_id: str = ""
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)
    ended_at: float | None = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    error: str | None = None

    @property
    def latency_ms(self) -> float:
        end = self.ended_at or time.monotonic()
        return round((end - self.started_at) * 1000, 2)


class Observability:
    """链路追踪与成本统计。

    - 无条件本地采集（内存 + 回调），用于评测与成本核算。
    - 配置 LANGCHAIN_API_KEY 时异步上报 LangSmith（OpenAI 兼容端点）。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._spans: list[Span] = []
        self._lock = asyncio.Lock()
        self._langsmith_client = None
        self._enabled = bool(settings.langsmith_api_key)

    def start_span(
        self, name: str, task_id: str, run_id: str = "", parent_run_id: str = "", **inputs: Any
    ) -> Span:
        span = Span(name=name, task_id=task_id, run_id=run_id, parent_run_id=parent_run_id, inputs=inputs)
        return span

    async def end_span(self, span: Span, outputs: dict[str, Any] | None = None, error: str | None = None) -> None:
        span.ended_at = time.monotonic()
        span.outputs = outputs or {}
        span.error = error
        async with self._lock:
            self._spans.append(span)
        if self._enabled:
            try:
                await self._report_langsmith(span)
            except Exception:
                pass  # 上报失败不影响主流程

    async def _report_langsmith(self, span: Span) -> None:
        import httpx

        if self._langsmith_client is None:
            self._langsmith_client = httpx.AsyncClient(
                base_url=self._settings.langsmith_endpoint,
                headers={
                    "x-api-key": self._settings.langsmith_api_key,
                    "Content-Type": "application/json",
                },
                timeout=5.0,
            )
        payload = {
            "name": span.name,
            "run_type": "chain",
            "inputs": span.inputs,
            "outputs": span.outputs,
            "start_time": span.started_at,
            "end_time": span.ended_at,
            "extra": {"metadata": {"task_id": span.task_id, "error": span.error}},
        }
        await self._langsmith_client.post(f"/runs/{span.run_id or 'new'}", json=payload)

    async def task_cost(self, task_id: str) -> tuple[int, float]:
        """返回 (总 token, 总成本元)。"""
        async with self._lock:
            tokens = sum(s.token_usage.total_tokens for s in self._spans if s.task_id == task_id)
            cost = sum(
                s.token_usage.cost(self._settings)
                for s in self._spans
                if s.task_id == task_id
            )
        return tokens, round(cost, 4)

    async def snapshot(self) -> dict[str, Any]:
        """评测用快照：各 span 延迟、失败数。"""
        async with self._lock:
            total = len(self._spans)
            failed = sum(1 for s in self._spans if s.error)
            avg_latency = (
                round(sum(s.latency_ms for s in self._spans) / total, 2) if total else 0.0
            )
            return {"spans": total, "failed": failed, "avg_latency_ms": avg_latency}

    async def close(self) -> None:
        if self._langsmith_client is not None:
            try:
                await self._langsmith_client.aclose()
            except Exception:
                pass
            self._langsmith_client = None
