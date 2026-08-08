"""LLM 抽象层：OpenAI 兼容协议客户端 + MockLLM 离线降级。

- ChatLLM：真实模型（DeepSeek / 任意 OpenAI 兼容网关），支持 Function Calling。
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

import httpx

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
    """OpenAI 兼容协议客户端（DeepSeek 等）。"""

    name = "chat"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    @staticmethod
    def _to_openai_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [{"type": "function", "function": t} for t in tools]

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, list[dict[str, Any]] | None, TokenUsage]:
        payload: dict[str, Any] = {
            "model": self._settings.llm_model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature if temperature is not None else self._settings.llm_temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._settings.llm_max_tokens,
            "stream": False,
        }
        oai_tools = self._to_openai_tools(tools)
        if oai_tools:
            payload["tools"] = oai_tools
            payload["tool_choice"] = "auto"

        try:
            resp = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 请求失败: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"LLM 返回 {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        usage = data.get("usage", {})
        return (
            msg.get("content") or "",
            msg.get("tool_calls"),
            TokenUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


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
            # 模拟模型决策：调用全部工具（每个一次）
            tool_calls = [
                {
                    "id": f"mock_call_{i}",
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "arguments": json.dumps({"query": last_user[:80]}, ensure_ascii=False),
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
