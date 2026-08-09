"""ForumEngine：轻量协作调度器。

职责：
- 并行调度 Query / Media / Insight 三类 Agent（支持 5+ Agent 协作，统一上下文协议）；
- 统一采集各 Agent 运行日志，结构化聚合分析结果；
- 支持指定 Agent 子集、串行/并行两种模式；
- 输出聚合摘要（供 ReportEngine 成稿）。
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable

from ..agents.agents import build_agents
from ..agents.base import AgentContext, BaseAgent
from ..agents.tools import ToolRegistry, build_default_registry
from ..config import Settings
from ..core.llm import BaseLLM
from ..core.observability import Observability

# 进度回调：stage(str), percent(int, 0-100)
ProgressFn = Callable[[str, int], Any]


class ForumEngine:
    """论坛式协作调度器：并行调度 Agent，统一日志与结果聚合。"""

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry | None = None,
        observability: Observability | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._tools = tools or build_default_registry()
        self._observability = observability
        self._settings = settings
        self._agents = build_agents(llm, self._tools, settings)
        # 已采集的运行日志（结构化）
        self.logs: list[dict[str, Any]] = []

    @property
    def agent_names(self) -> list[str]:
        return list(self._agents)

    def get_agent(self, name: str) -> BaseAgent:
        return self._agents[name]

    async def run(
        self,
        query: str,
        task_id: str = "task_demo",
        agents: list[str] | None = None,
        parallel: bool = True,
        progress: ProgressFn | None = None,
    ) -> dict[str, Any]:
        """执行一次协作分析。

        :param agents: 参与 Agent 列表（默认全部：query/media/insight）
        :param parallel: True=并行调度，False=串行
        """
        names = agents or list(self._agents)
        ctx = AgentContext(task_id=task_id, query=query)
        started = time.monotonic()

        if progress:
            progress("调度", 5)

        async def _run_one(name: str) -> tuple[str, dict[str, Any] | None]:
            """执行单个 Agent：支持 Agent 级故障注入与自动重试（健壮性）。"""
            span = None
            if self._observability:
                span = self._observability.start_span(
                    f"agent:{name}", task_id=task_id, query=query
                )
            agent_retries = self._settings.agent_retries if self._settings else 1
            agent_fault_rate = self._settings.agent_fault_rate if self._settings else 0.0
            last_exc: Exception | None = None
            for attempt in range(agent_retries + 1):
                try:
                    if (
                        agent_fault_rate > 0
                        and random.random() < agent_fault_rate
                    ):
                        raise RuntimeError(f"注入 Agent 瞬时故障（第 {attempt + 1} 次尝试）")
                    prompt_before = ctx.usage.prompt_tokens
                    completion_before = ctx.usage.completion_tokens
                    result = await self._agents[name].run(ctx)
                    if span:
                        # 记录本次 Agent 的 token 增量（成本统计）
                        span.token_usage.prompt_tokens = ctx.usage.prompt_tokens - prompt_before
                        span.token_usage.completion_tokens = (
                            ctx.usage.completion_tokens - completion_before
                        )
                        span.token_usage.total_tokens = (
                            span.token_usage.prompt_tokens + span.token_usage.completion_tokens
                        )
                        await self._observability.end_span(span, outputs={"summary": result})
                    if attempt > 0:
                        ctx.log(name, "agent_retry", f"{name} 第 {attempt + 1} 次执行成功")
                    return name, result
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < agent_retries:
                        ctx.log(name, "agent_retry", f"{name} 执行失败，重试: {exc}")
            if span:
                await self._observability.end_span(span, error=str(last_exc))
            return name, None

        if parallel:
            results = await asyncio.gather(*(_run_one(n) for n in names))
        else:
            results = []
            for i, n in enumerate(names):
                if progress:
                    progress(f"{n} 执行中", 10 + int(i / len(names) * 70))
                results.append(await _run_one(n))

        if progress:
            progress("聚合", 80)

        # 结构化聚合
        agent_results: dict[str, dict[str, Any]] = {}
        failed: list[str] = []
        for name, result in results:
            if result is None:
                failed.append(name)
            else:
                agent_results[name] = result

        # 汇总日志（统一采集）
        self.logs = list(ctx.logs)
        summary = self._aggregate(ctx, agent_results, failed, time.monotonic() - started)

        if progress:
            progress("完成", 100)
        return summary

    def _aggregate(
        self,
        ctx: AgentContext,
        agent_results: dict[str, dict[str, Any]],
        failed: list[str],
        elapsed: float,
    ) -> dict[str, Any]:
        return {
            "task_id": ctx.task_id,
            "query": ctx.query,
            "agent_results": agent_results,
            "failed_agents": failed,
            "success_rate": round(
                len(agent_results) / max(1, len(agent_results) + len(failed)) * 100, 1
            ),
            "elapsed_seconds": round(elapsed, 2),
            "total_tokens": ctx.usage.total_tokens,
            "logs": ctx.logs,
        }
