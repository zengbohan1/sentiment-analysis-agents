"""故障注入对比评测：任务完成率 / 工具调用成功率在「无防护」与「重试防护」下的对比。

用法（离线 MockLLM）：
    # Agent 级故障（模拟 LLM 调用失败）：基线 vs 防护
    python scripts/bench_fault.py --level agent --fault-rate 0.04 --agent-retries 0 --out results/fault_agent_baseline.json
    python scripts/bench_fault.py --level agent --fault-rate 0.04 --out results/fault_agent_retry.json
    # 工具级故障：基线 vs 防护
    python scripts/bench_fault.py --level tool --fault-rate 0.04 --tool-retries 0 --out results/fault_tool_baseline.json
    python scripts/bench_fault.py --level tool --fault-rate 0.04 --out results/fault_tool_retry.json

输出：完成率 / 工具调用成功率 / 失败重试次数。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.core.llm import build_llm  # noqa: E402
from app.core.observability import Observability  # noqa: E402
from app.engines.forum_engine import ForumEngine  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUERIES = [
    "分析新品手机「星云 X1」近期的口碑舆情，关注产品体验、价格、续航和售后",
    "分析奶茶品牌联名活动的传播效果与网友评价",
    "分析车企召回事件的舆情风险与应对建议",
    "分析景区门票涨价争议的舆论走向",
    "分析游戏版本更新后的玩家反馈",
    "分析直播带货翻车事件的影响范围",
]


def load_queries(n: int) -> list[str]:
    """复用 60 条评测集的前 n 条 query（无评测集则循环内置）。"""
    eval_path = PROJECT_ROOT / "data/eval/eval_set.jsonl"
    if eval_path.exists():
        queries = []
        with open(eval_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    queries.append(json.loads(line)["query"])
                    if len(queries) >= n:
                        break
        if queries:
            return queries
    return [QUERIES[i % len(QUERIES)] for i in range(n)]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["agent", "tool"], default="agent")
    parser.add_argument("--fault-rate", type=float, default=0.04)
    parser.add_argument("--agent-retries", type=int, default=1)
    parser.add_argument("--tool-retries", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--out", default="results/fault_report.json")
    args = parser.parse_args()

    if args.level == "agent":
        settings = Settings(
            llm_mock=True,
            agent_fault_rate=args.fault_rate,
            agent_retries=args.agent_retries,
        )
    else:
        settings = Settings(
            llm_mock=True,
            fault_inject_rate=args.fault_rate,
            tool_retries=args.tool_retries,
        )
    llm = build_llm(settings)
    obs = Observability(settings)
    fe = ForumEngine(llm, observability=obs, settings=settings)

    queries = load_queries(args.tasks)
    retry_label = args.agent_retries if args.level == "agent" else args.tool_retries
    print(f"故障注入评测：{len(queries)} 条 | 级别={args.level} | 注入率 {args.fault_rate:.0%} | "
          f"重试 {retry_label} 次 | LLM={llm.name}")

    results = []
    started = time.monotonic()
    for i, q in enumerate(queries):
        tid = f"fault_{i}"
        summary = await fe.run(q, task_id=tid)
        logs = summary["logs"]
        tool_ok = sum(1 for lg in logs if lg.get("level") == "tool_ok")
        tool_fail = sum(1 for lg in logs if lg.get("level") == "tool_fail")
        tool_retries = sum(1 for lg in logs if lg.get("level") == "tool_retry")
        agent_retries = sum(1 for lg in logs if lg.get("level") == "agent_retry")
        results.append({
            "id": tid,
            "success": summary["success_rate"] >= 100.0,
            "success_rate": summary["success_rate"],
            "failed_agents": summary["failed_agents"],
            "tool_calls": tool_ok + tool_fail,
            "tool_ok": tool_ok,
            "tool_fail": tool_fail,
            "tool_retries": tool_retries,
            "agent_retries": agent_retries,
        })

    total = len(results)
    ok = sum(1 for r in results if r["success"])
    tool_total = sum(r["tool_calls"] for r in results)
    tool_fail_total = sum(r["tool_fail"] for r in results)
    tool_retry_total = sum(r["tool_retries"] for r in results)
    agent_retry_total = sum(r["agent_retries"] for r in results)

    report = {
        "level": args.level,
        "fault_inject_rate": args.fault_rate,
        "agent_retries": args.agent_retries,
        "tool_retries": args.tool_retries,
        "total": total,
        "completed": ok,
        "completion_rate": round(ok / total * 100, 1),
        "tool_calls": tool_total,
        "tool_failures": tool_fail_total,
        "tool_success_rate": round(
            (tool_total - tool_fail_total) / max(1, tool_total) * 100, 1
        ),
        "tool_retry_count": tool_retry_total,
        "agent_retry_count": agent_retry_total,
        "wall_seconds": round(time.monotonic() - started, 2),
    }
    out_path = args.out
    if not Path(out_path).is_absolute():
        out_path = str(PROJECT_ROOT / out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"完成率 {report['completion_rate']}%（{ok}/{total}）｜"
          f"工具成功率 {report['tool_success_rate']}%（失败 {tool_fail_total}/{tool_total}）｜"
          f"工具重试 {tool_retry_total} 次｜Agent 重试 {agent_retry_total} 次")
    print(f"报告已保存：{out_path}")

    await obs.close()
    await llm.aclose()


if __name__ == "__main__":
    asyncio.run(main())
