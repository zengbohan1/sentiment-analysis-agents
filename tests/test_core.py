"""核心模块测试：LLM、缓存、工具、Agent、ForumEngine、ReportEngine、TaskManager。"""
from __future__ import annotations

import asyncio
import os

import pytest

from app.agents.agents import build_agents
from app.agents.base import AgentContext
from app.agents.tools import build_default_registry
from app.config import Settings
from app.core.llm import ChatLLM, LLMMessage, MockLLM, TokenUsage
from app.core.task_manager import FAILED, RUNNING, SUCCEEDED


# ---------- LLM ----------

@pytest.mark.asyncio
async def test_mock_llm_returns_deterministic_reply():
    llm = MockLLM()
    reply, tool_calls, usage = await llm.complete([LLMMessage(role="user", content="测试舆情")])
    assert reply
    assert usage.total_tokens > 0


def test_token_usage_cost():
    settings = type("S", (), {"token_cost_input_per_1k": 0.001, "token_cost_output_per_1k": 0.002})()
    u = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
    assert u.total_tokens == 2000
    assert u.cost(settings) == pytest.approx(0.003)


# ---------- ChatLLM（LangChain 接入） ----------

def test_chat_llm_message_conversion():
    """LLMMessage -> langchain messages（含 assistant 工具调用与 tool 回填）。"""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    msgs = [
        LLMMessage(role="system", content="你是分析师"),
        LLMMessage(role="user", content="分析口碑"),
        LLMMessage(
            role="assistant",
            content="正在调用工具",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_posts", "arguments": '{"keyword": "口碑"}'},
                }
            ],
        ),
        LLMMessage(role="tool", content='{"total": 1}', tool_call_id="call_1", name="search_posts"),
    ]
    lc = ChatLLM._to_lc_messages(msgs)
    assert isinstance(lc[0], SystemMessage)
    assert isinstance(lc[1], HumanMessage)
    assert isinstance(lc[2], AIMessage)
    assert lc[2].tool_calls[0]["name"] == "search_posts"
    assert lc[2].tool_calls[0]["args"] == {"keyword": "口碑"}
    assert isinstance(lc[3], ToolMessage)
    assert lc[3].tool_call_id == "call_1"


def test_chat_llm_tool_calls_roundtrip():
    """LangChain tool_calls 格式与 Agent 层 OpenAI 格式双向转换。"""
    lc_calls = [
        {"id": "call_x", "name": "query_db", "args": {"table": "posts"}, "type": "tool_call"}
    ]
    oai = ChatLLM._from_lc_tool_calls(lc_calls)
    assert oai[0]["function"]["name"] == "query_db"
    assert '"table": "posts"' in oai[0]["function"]["arguments"]
    back = ChatLLM._to_lc_tool_calls(oai)
    assert back[0]["name"] == "query_db"
    assert back[0]["args"] == {"table": "posts"}
    assert back[0]["id"] == "call_x"


@pytest.mark.asyncio
async def test_chat_llm_complete_with_langchain_model():
    """complete() 通过 LangChain 模型返回文本 / 工具调用 / 用量（mock 模型）。"""
    from langchain_core.messages import AIMessage

    class FakeModel:
        def bind_tools(self, tools, **kwargs):
            assert tools[0]["function"]["name"] == "search_posts"
            return self

        async def ainvoke(self, messages, **kwargs):
            assert messages[0].type == "system"
            assert messages[-1].type == "human"
            return AIMessage(
                content="",
                tool_calls=[{"id": "c1", "name": "search_posts", "args": {"keyword": "口碑"}}],
                usage_metadata={"input_tokens": 12, "output_tokens": 8, "total_tokens": 20},
            )

    llm = ChatLLM(Settings(llm_api_key="sk-test", llm_base_url="https://api.deepseek.com/v1"))
    llm._model = FakeModel()
    text, tool_calls, usage = await llm.complete(
        [LLMMessage(role="system", content="分析师"), LLMMessage(role="user", content="口碑")],
        tools=[{"name": "search_posts", "description": "检索帖子", "parameters": {}}],
    )
    assert text == ""
    assert tool_calls[0]["function"]["name"] == "search_posts"
    assert tool_calls[0]["function"]["arguments"] == '{"keyword": "口碑"}'
    assert usage.total_tokens == 20


@pytest.mark.asyncio
async def test_chat_llm_complete_content_only():
    """模型直接返回文本（无工具调用）时链路正常。"""
    from langchain_core.messages import AIMessage

    class FakeModel:
        async def ainvoke(self, messages, **kwargs):
            return AIMessage(
                content="分析完成",
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            )

    llm = ChatLLM(Settings(llm_api_key="sk-test", llm_base_url="https://api.deepseek.com/v1"))
    llm._model = FakeModel()
    text, tool_calls, usage = await llm.complete([LLMMessage(role="user", content="测试")])
    assert text == "分析完成"
    assert tool_calls is None
    assert usage.total_tokens == 30


# ---------- 工具 ----------

def test_tool_registry_has_four_tools():
    reg = build_default_registry()
    assert set(reg.names()) == {"search_posts", "parse_document", "query_db", "sentiment_analysis"}


@pytest.mark.asyncio
async def test_search_tool_filters_by_keyword():
    reg = build_default_registry()
    fn = reg.get("search_posts")
    out = await fn(keyword="微博", limit=5)
    import json

    data = json.loads(out)
    assert data["total"] >= 1
    assert all(p["platform"] == "微博" for p in data["posts"])


# ---------- Agent ----------

@pytest.mark.asyncio
async def test_three_agents_run_with_unified_context(mock_llm):
    reg = build_default_registry()
    agents = build_agents(mock_llm, reg)
    assert set(agents) == {"query", "media", "insight"}
    ctx = AgentContext(task_id="t1", query="分析新品口碑")
    for name, agent in agents.items():
        result = await agent.run(ctx)
        assert result["agent"] == name
        assert ctx.shared[f"{name}_result"]["agent"] == name
    # 统一上下文：日志被采集
    assert len(ctx.logs) >= 3


# ---------- ForumEngine ----------

@pytest.mark.asyncio
async def test_forum_engine_parallel_aggregation(forum):
    summary = await forum.run("分析星云 X1 手机口碑", task_id="task-1")
    assert set(summary["agent_results"]) == {"query", "media", "insight"}
    assert summary["success_rate"] == 100.0
    assert summary["failed_agents"] == []
    assert summary["elapsed_seconds"] >= 0
    assert len(summary["logs"]) >= 3


@pytest.mark.asyncio
async def test_forum_engine_subset_and_serial(forum):
    summary = await forum.run("测试", task_id="task-2", agents=["query", "insight"], parallel=False)
    assert set(summary["agent_results"]) == {"query", "insight"}


@pytest.mark.asyncio
async def test_forum_engine_progress_callback(forum):
    stages = []

    def cb(stage, percent):
        stages.append((stage, percent))

    await forum.run("测试进度", task_id="task-3", progress=cb)
    assert stages[0][1] == 5
    assert stages[-1][1] == 100


# ---------- ReportEngine ----------

@pytest.mark.asyncio
async def test_report_engine_generates_three_formats(report_engine, tmp_path):
    summary = {
        "task_id": "task-r1",
        "query": "测试",
        "agent_results": {"query": {"agent": "query", "facts": "事实一"}},
        "failed_agents": [],
        "success_rate": 100.0,
        "elapsed_seconds": 1.2,
        "total_tokens": 10,
        "logs": [],
    }
    paths = await report_engine.generate(summary, str(tmp_path))
    for fmt in ("html", "markdown", "pdf"):
        assert os.path.exists(paths[fmt])
        assert os.path.getsize(paths[fmt]) > 0


# ---------- TaskManager ----------

@pytest.mark.asyncio
async def test_task_manager_submit_and_succeed(task_manager):
    tid = await task_manager.submit("分析新品口碑")
    for _ in range(100):
        data = await task_manager.get_task(tid)
        if data and data["status"] == SUCCEEDED:
            break
        await asyncio.sleep(0.02)
    data = await task_manager.get_task(tid)
    assert data["status"] == SUCCEEDED
    assert data["progress"] == 100
    assert data["report_paths"]["html"]
    assert os.path.exists(data["report_paths"]["html"])


@pytest.mark.asyncio
async def test_task_manager_sse_stream(task_manager):
    tid = await task_manager.submit("SSE 测试")
    events = []
    async for payload in _collect(task_manager, tid, timeout=10):
        events.append(payload)
    assert events[-1]["status"] == SUCCEEDED
    # 断线重连：重新订阅立即收到当前快照
    q = task_manager.subscribe(tid)
    snapshot = await asyncio.wait_for(q.get(), timeout=2)
    assert snapshot["status"] == SUCCEEDED


async def _collect(tm, tid, timeout=10):
    q = tm.subscribe(tid)
    try:
        while True:
            payload = await asyncio.wait_for(q.get(), timeout=timeout)
            yield payload
            if payload.get("status") in (SUCCEEDED, FAILED):
                break
    finally:
        tm.unsubscribe(tid, q)


@pytest.mark.asyncio
async def test_task_manager_stats(task_manager):
    await task_manager.submit("统计测试")
    await asyncio.sleep(0.05)
    stats = await task_manager.stats()
    assert stats["total"] >= 1
    assert stats["concurrency_limit"] > 0
