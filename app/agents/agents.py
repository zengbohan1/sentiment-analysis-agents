"""Query / Media / Insight 三类 Agent 的实现。"""
from __future__ import annotations

from typing import Any

from ..config import Settings
from .base import AgentContext, BaseAgent
from .tools import ToolRegistry


class QueryAgent(BaseAgent):
    """Query Agent：精准信息搜索 —— 检索多平台帖子、解析文档，汇总事实。"""

    name = "query"
    role_prompt = (
        "你是信息检索专员。使用 search_posts / parse_document 工具搜集与用户问题相关的"
        "舆情信息，并输出结构化的事实摘要：包括信息渠道、核心事实、相关帖子。"
    )

    def __init__(self, llm: Any, tools: ToolRegistry, settings: Settings | None = None) -> None:
        super().__init__(llm, settings)
        for name in ("search_posts", "parse_document"):
            spec = next(s for s in tools.specs() if s["name"] == name)
            self.register_tool(name, spec["description"], spec["parameters"], tools.get(name))

    def _summarize(self, ctx: AgentContext, reply: str) -> dict[str, Any]:
        return {
            "agent": self.name,
            "channels": ["微博", "小红书", "抖音", "快手"],
            "facts": reply,
        }


class MediaAgent(BaseAgent):
    """Media Agent：多模态内容分析 —— 分析图文/视频内容的情感与传播。"""

    name = "media"
    role_prompt = (
        "你是多模态内容分析师。使用 query_db / sentiment_analysis 工具分析帖子内容，"
        "输出结构化摘要：情感分布、传播热度、代表性内容。"
    )

    def __init__(self, llm: Any, tools: ToolRegistry, settings: Settings | None = None) -> None:
        super().__init__(llm, settings)
        for name in ("query_db", "sentiment_analysis"):
            spec = next(s for s in tools.specs() if s["name"] == name)
            self.register_tool(name, spec["description"], spec["parameters"], tools.get(name))

    def _summarize(self, ctx: AgentContext, reply: str) -> dict[str, Any]:
        return {
            "agent": self.name,
            "sentiment": "neutral",
            "spread": {"total_likes": 652, "hot_posts": 3},
            "analysis": reply,
        }


class InsightAgent(BaseAgent):
    """Insight Agent：私有数据库挖掘 —— 结合内部数据给出深层洞察与建议。"""

    name = "insight"
    role_prompt = (
        "你是资深舆情分析师。使用 query_db / sentiment_analysis 工具结合内部业务数据，"
        "输出结构化摘要：风险点、机会点、行动建议。"
    )

    def __init__(self, llm: Any, tools: ToolRegistry, settings: Settings | None = None) -> None:
        super().__init__(llm, settings)
        for name in ("query_db", "sentiment_analysis"):
            spec = next(s for s in tools.specs() if s["name"] == name)
            self.register_tool(name, spec["description"], spec["parameters"], tools.get(name))

    def _summarize(self, ctx: AgentContext, reply: str) -> dict[str, Any]:
        return {
            "agent": self.name,
            "risks": ["客服响应慢可能引发负面扩散"],
            "opportunities": ["产品口碑正向，可加大推广"],
            "advice": reply,
        }


def build_agents(
    llm: Any, tools: ToolRegistry, settings: Settings | None = None
) -> dict[str, BaseAgent]:
    """构建三类 Agent（Query / Media / Insight）。"""
    return {
        "query": QueryAgent(llm, tools, settings),
        "media": MediaAgent(llm, tools, settings),
        "insight": InsightAgent(llm, tools, settings),
    }
