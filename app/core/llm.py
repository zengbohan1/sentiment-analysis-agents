"""LLM 抽象层：LangChain（ChatOpenAI）+ MockLLM 离线降级。

- ChatLLM：基于 langchain-openai 的 ChatOpenAI（DeepSeek / 任意 OpenAI 兼容网关），
  通过 bind_tools 支持 Function Calling 工具调用。
- MockLLM：无 API Key 时的确定性回复，保证离线可跑、可测试、可评测。
- 统一 TokenUsage 统计，接入可观测模块。

工具调用循环由 Agent 层驱动：
  1) complete(messages, tools) -> (text, tool_calls, usage)
  2) 有 tool_calls 时 Agent 执行工具并回填 tool 消息，继续调用
  3) 无 tool_calls 时结束
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from ..config import Settings


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def cost(self, settings: Settings) -> float:
        return (
            self.prompt_tokens / 1000 * settings.token_cost_input_per_1k
            + self.completion_tokens / 1000 * settings.token_cost_output_per_1k
        )


@dataclass
class LLMMessage:
    role: str  # system / user / assistant / tool
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


class LLMError(RuntimeError):
    """LLM 调用异常（网络 / 鉴权 / 限流等）。"""


class BaseLLM:
    """LLM 接口：单次完成，返回 (文本, 工具调用列表, 用量)。"""

    name: str = "base"

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, TokenUsage]:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


class ChatLLM(BaseLLM):
    """LangChain ChatOpenAI 客户端（DeepSeek / 任意 OpenAI 兼容网关）。"""

    name = "chat"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        from langchain_openai import ChatOpenAI

        self._model = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=120,
            max_retries=2,
        )

    # ---------- 消息转换 ----------

    @staticmethod
    def _to_lc_messages(messages: list[LLMMessage]) -> list[Any]:
        """LLMMessage -> langchain_core.messages（含工具调用回填消息）。"""
        out: list[Any] = []
        for m in messages:
            if m.role == "system":
                out.append(SystemMessage(content=m.content))
            elif m.role == "assistant":
                if m.tool_calls:
                    out.append(
                        AIMessage(
                            content=m.content,
                            tool_calls=ChatLLM._to_lc_tool_calls(m.tool_calls),
                        )
                    )
                else:
                    out.append(AIMessage(content=m.content))
            elif m.role == "tool":
                out.append(
                    ToolMessage(content=m.content, tool_call_id=m.tool_call_id, name=m.name)
                )
            else:
                out.append(HumanMessage(content=m.content))
        return out

    @staticmethod
    def _to_lc_tool_calls(oai_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """OpenAI 格式 tool_calls -> LangChain 格式（name/args/id）。"""
        return [
            {
                "name": c["function"]["name"],
                "args": json.loads(c["function"].get("arguments") or "{}"),
                "id": c.get("id") or f"call_{i}",
                "type": "tool_call",
            }
            for i, c in enumerate(oai_calls)
        ]

    @staticmethod
    def _from_lc_tool_calls(lc_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """LangChain 格式 tool_calls -> OpenAI 格式（Agent 层回填用）。"""
        return [
            {
                "id": c.get("id") or f"call_{i}",
                "type": "function",
                "function": {
                    "name": c["name"],
                    "arguments": json.dumps(c.get("args") or {}, ensure_ascii=False),
                },
            }
            for i, c in enumerate(lc_calls)
        ]

    # ---------- 调用 ----------

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, TokenUsage]:
        try:
            model = self._model
            if tools:
                # 绑定 Function Calling 工具（OpenAI 函数 schema）
                model = model.bind_tools([{"type": "function", "function": t} for t in tools])
            kwargs: dict[str, Any] = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            ai = await model.ainvoke(self._to_lc_messages(messages), **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"LLM 请求失败: {exc}") from exc

        content = ai.content
        if isinstance(content, list):  # 内容块（多模态响应）
            content = "".join(
                str(c.get("text", "")) if isinstance(c, dict) else str(c) for c in content
            )
        tool_calls = self._from_lc_tool_calls(ai.tool_calls) if ai.tool_calls else None
        um = ai.usage_metadata or {}
        return (
            content or "",
            tool_calls,
            TokenUsage(
                prompt_tokens=um.get("input_tokens", 0),
                completion_tokens=um.get("output_tokens", 0),
                total_tokens=um.get("total_tokens", 0),
            ),
        )

    async def aclose(self) -> None:
        try:
            await self._model.aclose()
        except Exception:  # noqa: BLE001
            pass


class MockLLM(BaseLLM):
    """确定性 Mock：无 Key / 离线 / 测试时使用。

    模拟完整工具调用循环：
    - 首次调用（消息中无 tool 结果）：声明调用所有已注册工具（每个至多一次）；
    - 工具执行结果回填后再次调用：输出确定性最终分析文本。
    """

    name = "mock"

    _FALLBACK_REPLY = "（无有效输入，无法分析）"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._call_count = 0

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, TokenUsage]:
        self._call_count += 1
        last_user = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user = m.content
                break

        has_tool_results = any(m.role == "tool" for m in messages)
        if tools and not has_tool_results:
            # 模拟模型决策：按工具 JSON Schema 生成参数，调用全部工具（每个一次）
            tool_calls = [
                {
                    "id": f"mock_call_{i}",
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "arguments": json.dumps(
                            self._args_from_schema(t, last_user), ensure_ascii=False
                        ),
                    },
                }
                for i, t in enumerate(tools)
            ]
            usage = TokenUsage(prompt_tokens=len(last_user) * 2 + 100, completion_tokens=80)
            return "正在调用工具获取信息。", tool_calls, usage

        reply = self._compose_final_reply(last_user)
        usage = TokenUsage(
            prompt_tokens=len(last_user) * 2 + 120,
            completion_tokens=len(reply) * 2 + 60,
        )
        return reply, None, usage

    @staticmethod
    def _args_from_schema(tool: dict[str, Any], query: str) -> dict[str, Any]:
        """按工具 JSON Schema 生成模拟参数（字符串参数填查询，其余按类型取样例值）。"""
        props = (tool.get("parameters") or {}).get("properties") or {}
        args: dict[str, Any] = {}
        for pname, pspec in props.items():
            ptype = pspec.get("type", "string")
            if ptype == "integer":
                args[pname] = 1
            elif ptype == "number":
                args[pname] = 0.0
            elif ptype == "boolean":
                args[pname] = True
            else:
                args[pname] = query[:80]
        return args

    def _compose_final_reply(self, query: str) -> str:
        if not query.strip():
            return self._FALLBACK_REPLY
        topic = re.sub(r"\s+", "", query)[:20]
        return (
            f"【Mock 分析】针对「{topic}」：整体舆情以中性偏正面为主，"
            "主要讨论集中在产品体验与价格；负面占比较低，建议关注服务类反馈。"
        )


def build_llm(settings: Settings | None = None) -> BaseLLM:
    """按配置构建 LLM：无 Key / mock 开关时返回 MockLLM。"""
    settings = settings or Settings()
    if settings.use_mock_llm:
        return MockLLM(settings)
    return ChatLLM(settings)
