"""Pydantic 请求/响应模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="舆情分析需求")
    agents: list[str] | None = Field(
        default=None, description="参与 Agent（query/media/insight），默认全部"
    )
    parallel: bool = Field(default=True, description="是否并行调度")
    title: str | None = Field(default=None, description="报告标题")


class TaskResponse(BaseModel):
    task_id: str
    status: str
    progress: int = 0
    stage: str = ""


class TaskDetail(BaseModel):
    task_id: str
    status: str
    progress: int
    stage: str
    error: str | None = None
    summary: dict[str, Any] | None = None
    report_paths: dict[str, str] | None = None
    total_tokens: int = 0
    total_cost: float = 0.0


class StatsResponse(BaseModel):
    total: int
    by_status: dict[str, int]
    concurrency_limit: int
