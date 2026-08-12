"""并发压测：50+ 任务并发、串行/并行调度 P50/P95/P99、报告生成耗时。

用法：
    # 并发压测（TaskManager 层，任务全链路，默认 MockLLM 快速验证）
    python scripts/bench.py --concurrency 60 --tasks 60

    # 调度模式对比（ForumEngine 层，真实 LLM 抽样）
    python scripts/bench.py --mode serial --tasks 6 --real
    python scripts/bench.py --mode parallel --tasks 6 --real

输出：完成率 / 耗时分位数（P50/P95/P99）/ 报告生成耗时。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.cache import Cache  # noqa: E402
from app.core.llm import build_llm  # noqa: E402
from app.core.observability import Observability  # noqa: E402
from app.core.task_manager import SUCCEEDED, TaskManager  # noqa: E402
from app.engines.forum_engine import ForumEngine  # noqa: E402
from app.engines.report_engine import ReportEngine  # noqa: E402

QUERIES = [
    "分析新品手机「星云 X1」近期的口碑舆情，关注产品体验、价格、续航和售后",
    "分析奶茶品牌联名活动的传播效果与网友评价",
    "分析车企召回事件的舆情风险与应对建议",
    "分析景区门票涨价争议的舆论走向",
    "分析游戏版本更新后的玩家反馈",
    "分析直播带货翻车事件的影响范围",
]


def _percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(len(values) * p))
    return round(values[idx], 2)


async def bench_tasks(concurrency: int, tasks: int, use_real: bool) -> dict:
    """TaskManager 层：并发提交任务，统计完成率与耗时分位数。

    MockLLM 模式（默认）：离线压测任务管控链路（队列/限流/SSE/报告生成）；
    --real：真实 LLM 全链路（并发会打真实 API，慎用）。
    """
    settings = get_settings()
    if not use_real:
        settings = settings.model_copy(update={"llm_mock": True})
    llm = build_llm(settings)
    cache = Cache(settings)
    obs = Observability(settings)
    await cache.connect()
    tm = TaskManager(settings, llm, cache, obs)

    started = time.monotonic()
    ids = [await tm.submit(QUERIES[i % len(QUERIES)]) for i in range(tasks)]
    # 并发等待全部终态
    async def _wait(tid: str) -> float:
        t0 = time.monotonic()
        while True:
            data = await tm.get_task(tid)
            if data and data["status"] in (SUCCEEDED, "failed", "cancelled"):
                return time.monotonic() - t0
            await asyncio.sleep(0.02)

    latencies = await asyncio.gather(*[_wait(t) for t in ids])
    elapsed = time.monotonic() - started

    done = [await tm.get_task(t) for t in ids]
    ok = sum(1 for d in done if d and d["status"] == SUCCEEDED)

    result = {
        "mode": "tasks",
        "concurrency": concurrency,
        "tasks": tasks,
        "llm": llm.name,
        "completed": ok,
        "completion_rate": round(ok / tasks * 100, 1),
        "wall_seconds": round(elapsed, 2),
        "p50_seconds": _percentile(latencies, 0.50),
        "p95_seconds": _percentile(latencies, 0.95),
        "p99_seconds": _percentile(latencies, 0.99),
    }
    await cache.close()
    await obs.close()
    await llm.aclose()
    return result


async def bench_engine(parallel: bool, tasks: int, use_real: bool) -> dict:
    """ForumEngine 层：串行/并行调度对比 + 报告生成耗时精确统计。"""
    settings = get_settings()
    if not use_real:
        settings = settings.model_copy(update={"llm_mock": True})
    llm = build_llm(settings)
    obs = Observability(settings)
    fe = ForumEngine(llm, observability=obs)
    reports = ReportEngine(settings)

    latencies: list[float] = []
    report_ms_list: list[float] = []
    ok = 0
    for i in range(tasks):
        t0 = time.monotonic()
        summary = await fe.run(
            QUERIES[i % len(QUERIES)],
            task_id=f"bench_{'par' if parallel else 'ser'}_{i}",
            parallel=parallel,
        )
        latencies.append(time.monotonic() - t0)
        if summary["success_rate"] >= 100.0:
            ok += 1
        r0 = time.monotonic()
        await reports.generate(summary, out_dir="reports/bench")
        report_ms_list.append((time.monotonic() - r0) * 1000)

    result = {
        "mode": "parallel" if parallel else "serial",
        "tasks": tasks,
        "llm": llm.name,
        "completed": ok,
        "completion_rate": round(ok / tasks * 100, 1),
        "p50_seconds": _percentile(latencies, 0.50),
        "p95_seconds": _percentile(latencies, 0.95),
        "p99_seconds": _percentile(latencies, 0.99),
        "report_gen_p50_ms": round(statistics.median(report_ms_list), 1),
        "report_gen_p95_ms": _percentile(report_ms_list, 0.95),
    }
    await obs.close()
    await llm.aclose()
    return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=60)
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--mode", choices=["tasks", "serial", "parallel"], default="tasks")
    parser.add_argument("--real", action="store_true", help="使用真实 LLM（默认 MockLLM）")
    args = parser.parse_args()

    if args.mode == "tasks":
        result = await bench_tasks(args.concurrency, args.tasks, args.real)
    else:
        result = await bench_engine(args.mode == "parallel", args.tasks, args.real)

    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
