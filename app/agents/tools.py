"""4 类 Function Calling 工具：检索 / 文档解析 / 数据库查询 / 情感分析。

每个工具注册为 (描述, 参数 JSON Schema, 异步执行函数)，供三类 Agent 共享；
工具层负责执行真实逻辑（此处为可运行的轻量实现 + 内存数据源），
LLM 通过 tool_calls 选择工具，由 Agent 循环执行并把结果回填给模型。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

# 内置演示数据源（可替换为 MindSpider 抓取 / 数据库查询）
_DEMO_POSTS: list[dict[str, Any]] = [
    {"id": 1, "platform": "微博", "content": "产品很好用，物流也快，五星好评！", "likes": 120, "ts": "2026-08-01 10:00"},
    {"id": 2, "platform": "微博", "content": "客服态度差，等了一小时没人理，差评。", "likes": 45, "ts": "2026-08-01 11:00"},
    {"id": 3, "platform": "小红书", "content": "性价比一般，但颜值在线，可以入手。", "likes": 88, "ts": "2026-08-01 12:00"},
    {"id": 4, "platform": "抖音", "content": "实测三天，续航不错，就是有点重。", "likes": 210, "ts": "2026-08-01 13:00"},
    {"id": 5, "platform": "快手", "content": "价格比去年涨了不少，观望中。", "likes": 33, "ts": "2026-08-01 14:00"},
    {"id": 6, "platform": "微博", "content": "售后响应及时，问题很快解决了，满意。", "likes": 156, "ts": "2026-08-01 15:00"},
]


def _wrap(fn: Any) -> Any:
    async def _run(**kwargs: Any) -> str:
        try:
            result = await fn(**kwargs)
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    _run.__name__ = fn.__name__
    return _run


async def _search_tool(keyword: str = "", limit: int = 5) -> dict[str, Any]:
    """检索工具：按关键词匹配演示数据源，返回命中的帖子。"""
    kw = (keyword or "").strip()
    hits = [
        p for p in _DEMO_POSTS
        if not kw or kw in p["content"] or kw in p["platform"]
    ]
    return {"total": len(hits), "posts": hits[: max(1, min(int(limit), 20))]}


async def _doc_parse_tool(url: str = "", content: str = "") -> dict[str, Any]:
    """文档解析工具：解析 URL / 文本，提取正文与关键词。"""
    if url:
        # 演示：模拟抓取网页并解析
        text = f"【{url}】页面正文：这是一篇关于舆情的示例文章，包含事件描述与网友评论。"
    else:
        text = content or "（无内容）"
    # 简易中文关键词提取：按 2-gram 统计（先剔除标点）
    _punct = "，。！？、；：""''（）《》【】,.!?;:\"'()<>[] \t\n\r"
    cleaned = text.translate(str.maketrans("", "", _punct))
    grams: dict[str, int] = {}
    for i in range(max(0, len(cleaned) - 1)):
        g = cleaned[i : i + 2]
        grams[g] = grams.get(g, 0) + 1
    keywords = sorted(grams, key=grams.get, reverse=True)[:8]
    return {"url": url, "text": text[:500], "keywords": keywords}


async def _db_query_tool(table: str = "posts", condition: str = "") -> dict[str, Any]:
    """数据库查询工具：查询演示数据源（对应 MySQL 舆情落库表）。"""
    rows = _DEMO_POSTS
    if condition:
        rows = [p for p in rows if condition in json.dumps(p, ensure_ascii=False)]
    return {"table": table, "rows": rows[:10], "count": len(rows)}


async def _sentiment_tool(text: str = "") -> dict[str, Any]:
    """情感分析工具：基于词表的轻量情感打分（-1 ~ 1）。"""
    pos_words = ["好评", "满意", "不错", "快", "好", "五星", "解决", "颜值", "性价比", "续航", "喜欢"]
    neg_words = ["差评", "差", "贵", "涨", "慢", "不理", "差劲", "问题", "等了一小时", "重"]
    pos = sum(1 for w in pos_words if w in text)
    neg = sum(1 for w in neg_words if w in text)
    score = (pos - neg) / max(1, pos + neg)
    if score > 0.2:
        label = "正面"
    elif score < -0.2:
        label = "负面"
    else:
        label = "中性"
    return {"text": text[:200], "score": round(score, 3), "label": label}


class ToolRegistry:
    """工具注册中心：注册 4 类 Function Calling 工具供 Agent 使用。"""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}

    def register(self, name: str, description: str, schema: dict, fn: Any) -> None:
        self._tools[name] = {
            "description": description,
            "parameters": schema,
            "fn": _wrap(fn),
        }

    def specs(self) -> list[dict[str, Any]]:
        return [
            {"name": n, "description": t["description"], "parameters": t["parameters"]}
            for n, t in self._tools.items()
        ]

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Any | None:
        tool = self._tools.get(name)
        return tool["fn"] if tool else None


def build_default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        "search_posts",
        "检索多平台舆情帖子，按关键词过滤，返回命中的帖子列表。",
        {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "搜索关键词"},
                "limit": {"type": "integer", "description": "返回条数，默认 5"},
            },
            "required": [],
        },
        _search_tool,
    )
    reg.register(
        "parse_document",
        "解析网页/文本内容，提取正文与关键词。",
        {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "网页地址"},
                "content": {"type": "string", "description": "文本内容"},
            },
            "required": [],
        },
        _doc_parse_tool,
    )
    reg.register(
        "query_db",
        "查询舆情数据库（帖子表），支持按条件过滤。",
        {
            "type": "object",
            "properties": {
                "table": {"type": "string", "description": "表名，默认 posts"},
                "condition": {"type": "string", "description": "过滤条件字符串"},
            },
            "required": [],
        },
        _db_query_tool,
    )
    reg.register(
        "sentiment_analysis",
        "对一段文本进行情感分析，返回情感标签与分数。",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "待分析文本"},
            },
            "required": ["text"],
        },
        _sentiment_tool,
    )
    return reg
