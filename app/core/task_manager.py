"""TaskManager：长耗时任务异步管控。

- 任务排队：asyncio.Semaphore 限流，支持 50+ 任务并发；
- 实时进度推送：SSE 订阅（task_id → 订阅者队列）；
- 断线重连：客户端携带 task_id 重新订阅，可获取历史进度；
- 历史进度恢复：进度快照缓存在 Redis（降级内存），重启后仍可查询；
- 任务终止：取消未开始/进行中的任务。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator

from ..config import Settings
from .cache import Cache
from .llm import BaseLLM
from .observability import Observability
from ..engines.forum_engine import ForumEngine
from ..engines.report_engine import ReportEngine

# 任务状态
QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = (
    "queued", "running", "succeeded", "failed", "cancelled",
)


class TaskManager:
    """任务队列 + SSE 进度推送 + 历史恢复。"""

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM,
        cache: Cache,
        observability: Observability | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._cache = cache
        self._obs = observability
        self._forum = ForumEngine(llm, observability=observability, settings=settings)
        self._reports = ReportEngine(settings)
        self._semaphore = asyncio.Semaphore(settings.task_concurrency)
        # 进度缓存：task_id -> {status, progress, stage, ...}
        self._progress: dict[str, dict[str, Any]] = {}
        # SSE 订阅者：task_id -> set[asyncio.Queue]
        self._subscribers: dict[str, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    # ---------- 任务生命周期 ----------

    async def submit(self, query: str, task_id: str | None = None) -> str:
        """提交任务，返回 task_id。"""
        tid = task_id or f"task_{int(time.time() * 1000)}"
        await self._update(tid, {"status": QUEUED, "progress": 0, "stage": "排队中"})
        asyncio.create_task(self._execute(tid, query))
        return tid

    async def _execute(self, task_id: str, query: str) -> None:
        async with self._semaphore:
            try:
                await self._update(task_id, {"status": RUNNING, "progress": 5, "stage": "调度中"})

                def _progress(stage: str, percent: int) -> None:
                    self._publish(task_id, {"status": RUNNING, "progress": percent, "stage": stage})

                # ForumEngine 并行调度（占 5% ~ 85%）
                summary = await self._forum.run(
                    query, task_id=task_id, progress=_progress
                )
                await self._update(task_id, {"progress": 88, "stage": "生成报告中"})

                # ReportEngine 生成 HTML/MD/PDF
                paths = await self._reports.generate(summary, out_dir="reports")

                # 成本统计（可观测）
                tokens, cost = 0.0, 0.0
                if self._obs:
                    tokens, cost = await self._obs.task_cost(task_id)
                else:
                    tokens = summary.get("total_tokens", 0)

                final = {
                    "status": SUCCEEDED,
                    "progress": 100,
                    "stage": "完成",
                    "summary": summary,
                    "report_paths": paths,
                    "total_tokens": tokens,
                    "total_cost": cost,
                }
                await self._update(task_id, final)
                self._publish(task_id, final)
            except asyncio.CancelledError:
                await self._update(task_id, {"status": CANCELLED, "stage": "已取消"})
                self._publish(task_id, {"status": CANCELLED, "stage": "已取消"})
            except Exception as exc:  # noqa: BLE001
                await self._update(task_id, {"status": FAILED, "stage": "失败", "error": str(exc)})
                self._publish(task_id, {"status": FAILED, "stage": "失败", "error": str(exc)})

    async def cancel(self, task_id: str) -> bool:
        """取消任务：仅对 queued/running 有效。"""
        async with self._lock:
            cur = self._progress.get(task_id)
            if cur is None or cur["status"] not in (QUEUED, RUNNING):
                return False
            cur["status"] = CANCELLED
            self._publish(task_id, {"status": CANCELLED, "stage": "取消中"})
            return True

    # ---------- 查询 / SSE ----------

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        """查询任务进度（断线重连 / 历史恢复入口）。"""
        async with self._lock:
            cur = self._progress.get(task_id)
        if cur is not None:
            return dict(cur)
        # 缓存兜底（跨进程/重启后）
        cached = await self._cache.get(f"task:{task_id}")
        return cached

    def subscribe(self, task_id: str) -> asyncio.Queue:
        """SSE 订阅：客户端断线后可重新订阅，立即收到当前快照。"""
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(task_id, set()).add(q)
        # 立即推送当前快照（断线重连恢复）
        cur = self._progress.get(task_id)
        if cur:
            q.put_nowait(dict(cur))
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue) -> None:
        subs = self._subscribers.get(task_id)
        if subs:
            subs.discard(q)
            if not subs:
                self._subscribers.pop(task_id, None)

    def _publish(self, task_id: str, payload: dict[str, Any]) -> None:
        subs = self._subscribers.get(task_id)
        if subs:
            for q in list(subs):
                try:
                    q.put_nowait(dict(payload))
                except asyncio.QueueFull:
                    pass

    async def _update(self, task_id: str, payload: dict[str, Any]) -> None:
        async with self._lock:
            cur = self._progress.get(task_id, {})
            cur.update(payload)
            self._progress[task_id] = cur
        # 持久化快照到缓存（供历史恢复）
        await self._cache.set(f"task:{task_id}", cur, ttl=86400)

    # ---------- 统计 ----------

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            statuses: dict[str, int] = {}
            for t in self._progress.values():
                statuses[t["status"]] = statuses.get(t["status"], 0) + 1
        return {
            "total": len(self._progress),
            "by_status": statuses,
            "concurrency_limit": self._settings.task_concurrency,
        }


async def sse_events(task_id: str, tm: TaskManager) -> AsyncIterator[str]:
    """SSE 事件生成器：推送任务进度，直到终态。"""
    q = tm.subscribe(task_id)
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield ":\n\n"  # 心跳保活
                continue
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if payload.get("status") in (SUCCEEDED, FAILED, CANCELLED):
                break
    finally:
        tm.unsubscribe(task_id, q)
