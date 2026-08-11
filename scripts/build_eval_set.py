"""生成 60 条评测集（JSONL）：覆盖 6 类舆情场景 × 10 条。

评测字段：query（输入）、topic（主题）、expected_aspects（应覆盖的要点）、
expected_agents（期望参与的 Agent）、completion_criteria（完成判据）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

TOPICS = [
    {
        "topic": "新品手机口碑",
        "aspects": ["产品体验", "价格", "续航", "售后服务"],
        "query_tpl": "分析新品手机「星云 X1」近期的口碑舆情，关注产品体验、价格、续航和售后。",
    },
    {
        "topic": "奶茶品牌联名",
        "aspects": ["联名热度", "网友态度", "消费意愿", "负面风险"],
        "query_tpl": "评估奶茶品牌「茶语」与动漫 IP 联名活动引发的舆情，包括热度、态度与风险。",
    },
    {
        "topic": "车企召回事件",
        "aspects": ["事件传播", "官方回应", "车主情绪", "品牌影响"],
        "query_tpl": "梳理某车企新能源车型召回事件的舆情走向，关注传播路径、回应效果与车主情绪。",
    },
    {
        "topic": "景区涨价争议",
        "aspects": ["舆论争议点", "游客反应", "监管态度", "舆情建议"],
        "query_tpl": "分析某 5A 景区门票涨价引发的舆情争议，给出景区公关建议。",
    },
    {
        "topic": "游戏版本更新",
        "aspects": ["玩家反馈", "平衡性", "付费争议", "社区氛围"],
        "query_tpl": "分析热门游戏新版本更新后的玩家舆情，包括平衡性、付费与社区氛围。",
    },
    {
        "topic": "直播带货翻车",
        "aspects": ["事件还原", "平台责任", "消费者维权", "主播信誉"],
        "query_tpl": "分析某头部主播直播带货翻车事件的舆情，关注消费者维权与平台责任。",
    },
]

VARIATIONS = [
    "请重点分析其中的负面情绪与潜在风险。",
    "同时给出企业应对建议与机会点。",
    "关注传播渠道与代表性言论。",
    "结合点赞/热度数据评估影响范围。",
    "输出结构化结论，便于直接生成报告。",
    "简要分析即可，突出核心结论。",
    "对比正面与负面观点占比。",
    "给出未来一周舆情走势预判。",
    "关注是否涉及监管与合规风险。",
    "总结需要优先处理的三个问题。",
]


def build_eval_set(n: int = 60) -> list[dict]:
    """构建 n 条评测集（默认 60 = 6 主题 × 10 变体）。"""
    items: list[dict] = []
    idx = 1
    for topic in TOPICS:
        for i, var in enumerate(VARIATIONS):
            if len(items) >= n:
                break
            items.append(
                {
                    "id": f"eval-{idx:03d}",
                    "topic": topic["topic"],
                    "query": f"{topic['query_tpl']}{var}",
                    "expected_aspects": topic["aspects"],
                    "expected_agents": ["query", "media", "insight"],
                    "completion_criteria": "summary 非空且 success_rate >= 100（三 Agent 全部成功）",
                }
            )
            idx += 1
        if len(items) >= n:
            break
    return items


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "data" / "eval" / "eval_set.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    items = build_eval_set(60)
    with open(out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"评测集已生成：{len(items)} 条 -> {out}")


if __name__ == "__main__":
    main()
