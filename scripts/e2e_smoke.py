"""端到端冒烟验证：真实 LLM（LangChain ChatOpenAI）全链路。

ForumEngine 并行调度 → Agent 工具调用循环 → 报告生成（HTML/MD/PDF）。
输出：耗时 / 工具调用次数与成功率 / token 用量与成本。

用法：.venv/Scripts/python scripts/e2e_smoke.py [--query "..."] [--agents query,media,insight]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.core.llm import build_llm
from app.core.observability import Observability
from app.engines.forum_engine import ForumEngine
from app.engines.report_engine import ReportEngine


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="分析新品手机「星云 X1」近期的口碑舆情，关注产品体验、价格、续航和售后")
    parser.add_argument("--agents", default="query,media,insight")
    parser.add_argument("--out-dir", default="reports/smoke")
    args = parser.parse_args()

    settings = Settings()
    if settings.use_mock_llm:
        print("[警告] 当前为 MockLLM 模式（LLM_API_KEY 未配置），走离线链路")

    obs = Observability(settings)
    llm = build_llm(settings)
    engine = ForumEngine(llm, observability=obs)
    reports = ReportEngine(settings)

    started = time.monotonic()
    summary = await engine.run(
        args.query,
        task_id="e2e_smoke",
        agents=args.agents.split(","),
        parallel=True,
    )
    run_ms = (time.monotonic() - started) * 1000

    # 报告生成（计时）
    r0 = time.monotonic()
    paths = await reports.generate(summary, out_dir=args.out_dir, title="端到端冒烟报告")
    report_ms = (time.monotonic() - r0) * 1000

    # 工具调用统计：从统一上下文日志里数（tool_ok / tool_fail）
    logs = summary["logs"]
    tool_ok = sum(1 for lg in logs if lg.get("level") == "tool_ok")
    tool_fail = sum(1 for lg in logs if lg.get("level") == "tool_fail")
    total_calls = tool_ok + tool_fail

    tokens, cost = await obs.task_cost("e2e_smoke")
    snap = await obs.snapshot()

    print(json.dumps({
        "llm": llm.name,
        "run_ms": round(run_ms, 1),
        "report_ms": round(report_ms, 1),
        "success_rate": summary["success_rate"],
        "failed_agents": summary["failed_agents"],
        "tool_calls": total_calls,
        "tool_failures": tool_fail,
        "total_tokens": tokens,
        "cost_yuan": cost,
        "spans": snap["spans"],
        "reports": paths,
    }, ensure_ascii=False, indent=2))

    await llm.aclose()
    await obs.close()


if __name__ == "__main__":
    asyncio.run(main())
