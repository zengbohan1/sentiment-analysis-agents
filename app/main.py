"""FastAPI 入口：任务提交 / 进度查询 / SSE 实时推送 / 统计。

健康检查、任务提交、任务状态查询、SSE 订阅、任务取消、统计。
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from .config import get_settings
from .core.cache import Cache
from .core.llm import build_llm
from .core.observability import Observability
from .core.task_manager import TaskManager, sse_events
from .schemas import AnalysisRequest, StatsResponse, TaskDetail, TaskResponse

settings = get_settings()

_cache = Cache(settings)
_obs = Observability(settings)
_tm: TaskManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _tm
    await _cache.connect()
    llm = build_llm(settings)
    _tm = TaskManager(settings, llm, _cache, _obs)
    yield
    await _cache.close()
    await _obs.close()
    await llm.aclose()


app = FastAPI(title=settings.app_name, lifespan=lifespan)


def _tm_or_raise() -> TaskManager:
    if _tm is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return _tm


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "app": settings.app_name, "mock_llm": settings.use_mock_llm}


@app.post("/v1/tasks", response_model=TaskResponse, status_code=202)
async def submit_task(req: AnalysisRequest) -> TaskResponse:
    tm = _tm_or_raise()
    # 校验 Agent 名单
    if req.agents:
        valid = {"query", "media", "insight"}
        unknown = set(req.agents) - valid
        if unknown:
            raise HTTPException(status_code=422, detail=f"未知 Agent: {unknown}")
    task_id = await tm.submit(req.query)
    return TaskResponse(task_id=task_id, status="queued")


@app.get("/v1/tasks/{task_id}", response_model=TaskDetail)
async def get_task(task_id: str) -> TaskDetail:
    tm = _tm_or_raise()
    data = await tm.get_task(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskDetail(task_id=task_id, **data)


@app.get("/v1/tasks/{task_id}/events")
async def task_events(task_id: str) -> EventSourceResponse:
    """SSE 实时进度：断线重连后自动恢复当前快照。"""
    tm = _tm_or_raise()
    if await tm.get_task(task_id) is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return EventSourceResponse(sse_events(task_id, tm))


@app.post("/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> TaskDetail:
    tm = _tm_or_raise()
    ok = await tm.cancel(task_id)
    if not ok:
        data = await tm.get_task(task_id)
        if data is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return TaskDetail(task_id=task_id, **data)
    data = await tm.get_task(task_id)
    return TaskDetail(task_id=task_id, **(data or {"status": "cancelled"}))


@app.get("/v1/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    tm = _tm_or_raise()
    return StatsResponse(**await tm.stats())


@app.get("/v1/reports/{task_id}")
async def download_report(
    task_id: str, fmt: str = Query(default="html", pattern="^(html|md|pdf)$")
) -> FileResponse:
    tm = _tm_or_raise()
    data = await tm.get_task(task_id)
    if data is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    paths = data.get("report_paths") or {}
    key = {"html": "html", "md": "markdown", "pdf": "pdf"}[fmt]
    path = paths.get(key)
    if not path:
        raise HTTPException(status_code=404, detail="报告尚未生成")
    return FileResponse(path, filename=f"{task_id}.{fmt}")
