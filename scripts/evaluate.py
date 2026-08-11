"""评测脚本：在 60 条评测集上运行 ForumEngine，统计任务完成率 / 成本 / 延迟 / 工具调用成功率。

用法：
    python scripts/evaluate.py [--limit 60] [--parallel] [--out results/eval_report.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.core.cache import Cache  # noqa: E402
from app.core.llm import build_llm  # noqa: E402
from app.core.observability import Observability  # noqa: E402
from app.engines.forum_engine import ForumEngine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_eval_set(path: str) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


async def run_single(fe: ForumEngine, item: dict, obs: Observability) -> dict:
    task_id = f"eval_{item['id']}"
    started = time.monotonic()
    span = obs.start_span(f"eval:{item['id']}", task_id=task_id, topic=item["topic"])
    summary = await fe.run(item["query"], task_id=task_id)
    elapsed = time.monotonic() - started
    await obs.end_span(span, outputs={"success_rate": summary["success_rate"]})

    success = summary["success_rate"] >= 100.0 and bool(summary["agent_results"])
    logs = summary["logs"]
    tool_ok = sum(1 for lg in logs if lg.get("level") == "tool_ok")
    tool_fail = sum(1 for lg in logs if lg.get("level") == "tool_fail")
    tokens, cost = await obs.task_cost(task_id)
    return {
        "id": item["id"],
        "topic": item["topic"],
        "success": success,
        "success_rate": summary["success_rate"],
        "elapsed": round(elapsed, 2),
        "agents_ok": len(summary["agent_results"]),
        "failed_agents": summary["failed_agents"],
        "tool_calls": tool_ok + tool_fail,
        "tool_failures": tool_fail,
        "total_tokens": tokens,
        "cost_yuan": round(cost, 4),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--parallel", action="store_true", help="评测项并行执行")
    parser.add_argument("--out", default="results/eval_report.json")
    args = parser.parse_args()

    settings = get_settings()
    eval_path = settings.eval_set_path
    if not Path(eval_path).is_absolute():
        eval_path = str(PROJECT_ROOT / eval_path)
    items = load_eval_set(eval_path)[: args.limit]
    llm = build_llm(settings)
    cache = Cache(settings)
    obs = Observability(settings)
    await cache.connect()
    fe = ForumEngine(llm, observability=obs)

    print(f"评测开始：{len(items)} 条（LLM={llm.name}）")
    started = time.monotonic()
    if args.parallel:
        results = await asyncio.gather(*(run_single(fe, it, obs) for it in items))
    else:
        results = []
        for i, it in enumerate(items):
            results.append(await run_single(fe, it, obs))
            if (i + 1) % 10 == 0:
                print(f"  进度 {i + 1}/{len(items)}")

    total = len(results)
    ok = sum(1 for r in results if r["success"])
    avg_elapsed = round(sum(r["elapsed"] for r in results) / max(1, total), 2)
    avg_agents = round(sum(r["agents_ok"] for r in results) / max(1, total), 2)
    snapshot = await obs.snapshot()

    tool_total = sum(r["tool_calls"] for r in results)
    tool_fail = sum(r["tool_failures"] for r in results)
    tool_success_rate = round((tool_total - tool_fail) / max(1, tool_total) * 100, 1)
    total_tokens = sum(r["total_tokens"] for r in results)
    total_cost = round(sum(r["cost_yuan"] for r in results), 4)

    report = {
        "total": total,
        "completed": ok,
        "completion_rate": round(ok / max(1, total) * 100, 1),
        "avg_elapsed_seconds": avg_elapsed,
        "avg_agents_ok": avg_agents,
        "tool_calls": tool_total,
        "tool_failures": tool_fail,
        "tool_success_rate": tool_success_rate,
        "total_tokens": total_tokens,
        "total_cost_yuan": total_cost,
        "avg_cost_yuan": round(total_cost / max(1, total), 4),
        "observability": snapshot,
        "llm": llm.name,
        "total_elapsed_seconds": round(time.monotonic() - started, 2),
        "details": results,
    }
    out_path = args.out
    if not Path(out_path).is_absolute():
        out_path = str(PROJECT_ROOT / out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n评测完成：完成率 {report['completion_rate']}%（{ok}/{total}），"
          f"平均耗时 {avg_elapsed}s，平均 Agent 成功数 {avg_agents}")
    print(f"工具调用：{tool_total} 次，成功率 {tool_success_rate}%（失败 {tool_fail} 次）")
    print(f"成本：总 {total_cost} 元，单任务平均 {report['avg_cost_yuan']} 元（{total_tokens} tokens）")
    print(f"报告已保存：{out_path}")

    await cache.close()
    await obs.close()
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
