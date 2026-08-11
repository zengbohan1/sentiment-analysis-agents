"""Agent 基类与统一上下文协议。

上下文协议（AgentContext）：所有 Agent 通过统一的上下文对象交换信息，
包含任务查询、共享状态（并行调度结果聚合）、日志与用量统计。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..config import Settings
from ..core.llm import BaseLLM, LLMMessage, TokenUsage


@dataclass
class AgentContext:
    """统一上下文协议：跨 Agent 协作的数据契约。"""

    task_id: str
    query: str
    # 共享状态：各 Agent 写入的命名结果，ForumEngine 负责聚合
    shared: dict[str, Any] = field(default_factory=dict)
    # 运行日志（结构化）
    logs: list[dict[str, Any]] = field(default_factory=list)
    # 用量累计
    usage: TokenUsage = field(default_factory=TokenUsage)

    def log(self, agent: str, level: str, message: str, **extra: Any) -> None:
        self.logs.append(
            {"agent": agent, "level": level, "message": message, **extra}
        )


# Function Calling 工具签名：名称 -> (描述, 参数 JSON Schema, 异步执行函数)
ToolFn = Callable[..., Awaitable[Any]]


class BaseAgent:
    """所有 Agent 的基类：统一 prompt 装配、工具执行循环、日志与用量记录。

    健壮性：
    - 工具调用失败自动重试（settings.tool_retries 次），重试仍失败降级为
      错误结果回填模型（不中断任务）；
    - settings.fault_inject_rate > 0 时按概率注入瞬时故障（仅评测用）。
    """

    name: str = "base"
    role_prompt: str = "你是一个分析助手。"

    def __init__(self, llm: BaseLLM, settings: Settings | None = None) -> None:
        self._llm = llm
        self._tools: dict[str, tuple[str, dict, ToolFn]] = {}
        self._settings = settings or Settings()
        self._tool_retries = max(0, self._settings.tool_retries)
        self._fault_rate = min(1.0, max(0.0, self._settings.fault_inject_rate))

    def register_tool(self, name: str, description: str, schema: dict, fn: ToolFn) -> None:
        self._tools[name] = (description, schema, fn)

    def tool_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "description": desc,
                "parameters": schema,
            }
            for name, (desc, schema, _) in self._tools.items()
        ]

    def _build_messages(self, ctx: AgentContext) -> list[LLMMessage]:
        return [
            LLMMessage(role="system", content=self.role_prompt),
            LLMMessage(role="user", content=ctx.query),
        ]

    def _summarize(self, ctx: AgentContext, reply: str) -> dict[str, Any]:
        """子类覆写：把 LLM 回复转为结构化摘要。"""
        return {"agent": self.name, "conclusion": reply}

    async def run(self, ctx: AgentContext) -> dict[str, Any]:
        """执行 Agent：装配消息 → LLM 工具调用循环 → 返回结构化摘要。

        工具调用循环：LLM 返回 tool_calls 时执行对应工具并回填结果，
        直到模型不再请求工具（上限 12 轮，防死循环）。
        """
        ctx.log(self.name, "info", f"{self.name} 开始执行")
        messages = self._build_messages(ctx)
        total_usage = TokenUsage()
        try:
            for _ in range(12):
                reply, tool_calls, usage = await self._llm.complete(
                    messages, tools=self.tool_specs()
                )
                total_usage.prompt_tokens += usage.prompt_tokens
                total_usage.completion_tokens += usage.completion_tokens
                total_usage.total_tokens += usage.total_tokens
                if not tool_calls:
                    break
                # 执行工具并回填
                messages.append(
                    LLMMessage(role="assistant", content=reply, tool_calls=tool_calls)
                )
                for tc in tool_calls:
                    fn = tc["function"]
                    name, arguments = fn["name"], fn.get("arguments", "{}")
                    ctx.log(self.name, "tool", f"调用工具 {name}")
                    kwargs = json.loads(arguments) if isinstance(arguments, str) else arguments
                    result, ok, last_exc = await self._execute_tool_with_retry(ctx, name, kwargs)
                    if ok:
                        ctx.log(self.name, "tool_ok", f"工具 {name} 调用成功")
                    else:
                        result = json.dumps({"error": str(last_exc)}, ensure_ascii=False)
                        ctx.log(self.name, "tool_fail", f"工具 {name} 调用失败: {last_exc}")
                    messages.append(
                        LLMMessage(
                            role="tool",
                            content=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                            tool_call_id=tc["id"],
                            name=name,
                        )
                    )
            ctx.usage.prompt_tokens += total_usage.prompt_tokens
            ctx.usage.completion_tokens += total_usage.completion_tokens
            ctx.usage.total_tokens += total_usage.total_tokens
            summary = self._summarize(ctx, reply)
            ctx.shared[f"{self.name}_result"] = summary
            ctx.log(self.name, "info", f"{self.name} 完成，摘要: {json.dumps(summary, ensure_ascii=False)[:200]}")
            return summary
        except Exception as exc:  # noqa: BLE001
            ctx.log(self.name, "error", f"{self.name} 失败: {exc}")
            raise

    async def _execute_tool(self, name: str, kwargs: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"未注册的工具: {name}")
        _, _, fn = tool
        return await fn(**kwargs)

    async def _execute_tool_with_retry(
        self, ctx: AgentContext, name: str, kwargs: dict[str, Any]
    ) -> tuple[Any, bool, Exception | None]:
        """执行工具：故障注入 + 自动重试，返回 (结果, 是否成功, 最后的异常)。"""
        last_exc: Exception | None = None
        for attempt in range(self._tool_retries + 1):
            try:
                if self._fault_rate > 0 and random.random() < self._fault_rate:
                    raise RuntimeError(f"注入瞬时故障（第 {attempt + 1} 次尝试）")
                return await self._execute_tool(name, kwargs), True, None
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._tool_retries:
                    ctx.log(
                        self.name, "tool_retry",
                        f"工具 {name} 第 {attempt + 1} 次失败，重试: {exc}",
                    )
        return None, False, last_exc
