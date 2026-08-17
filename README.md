# 多智能体舆情分析系统 · Multi-agent Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-D71F00?style=flat-square)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![Redis](https://img.shields.io/badge/Redis-DC382D?style=flat-square&logo=redis&logoColor=white)
![SSE](https://img.shields.io/badge/SSE-4B8BBE?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

![Tool Success](https://img.shields.io/badge/Tool_Success-100%25-4C9F70?style=flat-square)
![Concurrency](https://img.shields.io/badge/60_Tasks-100%25-4C9F70?style=flat-square)
![Cost](https://img.shields.io/badge/Cost-0.054CNY%2Ftask-FF6F00?style=flat-square)

> **Multi-agent Sentiment Analysis** — parallel Query / Media / Insight agents with a custom scheduler, SSE long-task control and report pipeline (HTML / PDF / Markdown). Built with FastAPI · LangChain · SQLAlchemy. Measured: tool success **100%** (265 calls) · **60**-task concurrency · **¥0.054**/task.

面向舆情分析的 AI 多智能体系统：自动完成多源舆情抓取、深度分析与报告生成。

> 技术栈：Python 3.11、FastAPI、LangChain（ChatOpenAI / bind_tools）、SQLAlchemy（异步）、PostgreSQL / MySQL 双数据源、Redis、SSE、Docker
> 离线可运行：无 API Key 时自动降级 MockLLM，全功能可用

## 架构

```
用户提问
   │
   ▼
FastAPI (/v1/tasks)
   │
   ├─ TaskManager：任务排队 → 限流（Semaphore）→ 进度发布
   │                 ├─ SSE 实时推送（断线重连自动恢复快照）
   │                 └─ Redis 进度快照（历史恢复）
   ▼
ForumEngine（轻量协作调度器，自研）
   ├─ Query Agent   —— 检索 / 文档解析工具
   ├─ Media Agent   —— 数据库查询 / 情感分析工具      （并行调度）
   └─ Insight Agent —— 数据库查询 / 情感分析工具
   │        └─ 统一上下文协议（AgentContext）+ 结构化日志采集
   ▼
ReportEngine 报告流水线
   └─ 模板成稿 → HTML / Markdown / PDF 三格式导出
   ▼
评测与可观测：60 条评测集 + LangSmith 链路追踪（可选）+ token 成本统计
```

## 核心亮点

- **多 Agent 并行调度**：Query / Media / Insight 三类 Agent，统一上下文协议（AgentContext），支持 5+ Agent 协作（可扩展）；真实 LLM 实测并行调度将任务 P95 耗时由 66.2s 降至 30.3s（降低 54%）。
- **自研 ForumEngine 调度器**：轻量协作调度器，统一采集 Agent 运行日志并结构化聚合分析结果；设计**检索、文档解析、数据库查询、情感分析** 4 类 Function Calling 工具，真实 LLM 实测 265 次工具调用成功率 100%。
- **长耗时任务异步管控**：FastAPI + SSE 实现任务排队、实时进度推送、**断线重连与历史进度恢复**；Redis 缓存热点查询；Semaphore 限流，压测 60 任务并发全部完成（100%）。
- **ReportEngine 报告流水线**：聚合结果按模板自动成稿，支持 **HTML / PDF / Markdown** 导出，实测报告生成耗时 P50 23.5ms / P95 110ms。
- **健壮性防护**：工具调用失败自动重试 + Agent 级失败重试；故障注入评测（4% 注入率）实测任务完成率由 81.7% 提升至 98.3%，工具调用成功率由 95.3% 提升至 99.4%。
- **评测与可观测**：60 条评测集（6 主题 × 10 变体）持续评估 Agent 效果；LangSmith 链路追踪 + token 成本统计，真实 LLM 实测单任务平均成本 0.054 元。

## 快速开始

```bash
# 1. 环境
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# 2. 配置
copy .env.example .env        # 填入 LLM_API_KEY（可留空，自动用 MockLLM）

# 3. 起基础设施（可选：PostgreSQL + Redis；不启动自动降级 SQLite/内存）
docker compose up -d
# 离线模式只需把 .env 的 DATABASE_URL 改为 sqlite+aiosqlite:///./sentiment.db

# 4. 启动服务
.venv/Scripts/python -m uvicorn app.main:app --port 8000

# 5. 提交分析任务（真实 LLM 链路：LangChain ChatOpenAI + Function Calling）
curl -X POST http://localhost:8000/v1/tasks -H "Content-Type: application/json" \
  -d '{"query": "分析新品手机「星云 X1」近期的口碑舆情"}'

# 6. SSE 实时进度
curl -N http://localhost:8000/v1/tasks/<task_id>/events

# 7. 下载报告（HTML / md / pdf）
curl -o report.html http://localhost:8000/v1/reports/<task_id>?fmt=html
```

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/tasks` | 提交分析任务（202 返回 task_id） |
| GET | `/v1/tasks/{id}` | 查询任务状态 / 进度（断线重连、历史恢复） |
| GET | `/v1/tasks/{id}/events` | SSE 实时进度推送 |
| POST | `/v1/tasks/{id}/cancel` | 取消任务 |
| GET | `/v1/stats` | 任务统计 |
| GET | `/v1/reports/{id}?fmt=html\|md\|pdf` | 下载报告 |
| GET | `/health` | 健康检查 |

## 评测

```bash
# 生成 60 条评测集
.venv/Scripts/python scripts/build_eval_set.py

# 跑评测（并行）
.venv/Scripts/python scripts/evaluate.py --limit 60 --parallel
# 输出 results/eval_report.json：完成率 / 工具调用成功率 / 平均耗时 / 单任务成本 / 可观测快照

# 并发压测（TaskManager 层，离线 MockLLM）
.venv/Scripts/python scripts/bench.py --concurrency 60 --tasks 60
# 调度模式对比（真实 LLM）
.venv/Scripts/python scripts/bench.py --mode serial --tasks 6 --real
.venv/Scripts/python scripts/bench.py --mode parallel --tasks 6 --real

# 故障注入对比（完成率 / 工具成功率防护效果）
.venv/Scripts/python scripts/bench_fault.py --level agent --fault-rate 0.04 --agent-retries 0
.venv/Scripts/python scripts/bench_fault.py --level agent --fault-rate 0.04
```

### 实测结果（本机，真实 DeepSeek-chat）

| 指标 | 实测 |
|------|------|
| 评测完成率（真实 LLM，6 条抽样） | 100%（6/6），平均耗时 18.6s |
| 工具调用成功率（真实 LLM） | 100%（265 次调用，0 失败） |
| 单任务平均成本（真实 LLM） | 0.054 元（约 44.6k tokens/任务） |
| 任务并发（TaskManager 压测） | 60 并发全部完成（100%），P95 0.52s（MockLLM 离线链路） |
| 调度 P95（真实 LLM） | 串行 66.2s → 并行 30.3s（-54%） |
| 报告生成耗时 | P50 23.5ms / P95 110ms |
| 故障注入（4%）：任务完成率 | 无防护 81.7% → 重试防护 98.3% |
| 故障注入（4%）：工具成功率 | 无防护 95.3% → 重试防护 99.4% |
| 测试与覆盖率 | 16 用例全过，核心模块 64%（agent 层 80%~96%） |

## 目录结构

```
app/
├── main.py              # FastAPI 入口（任务提交 / SSE / 报告下载）
├── config.py            # pydantic-settings 配置
├── schemas.py           # 请求/响应模型
├── models.py            # SQLAlchemy ORM（Task / AgentRun / Report）
├── agents/
│   ├── base.py          # Agent 基类 + 统一上下文协议 + 工具执行循环
│   ├── agents.py        # Query / Media / Insight 三类 Agent
│   └── tools.py         # 4 类 Function Calling 工具 + 注册中心
├── engines/
│   ├── forum_engine.py  # ForumEngine 轻量协作调度器
│   └── report_engine.py # ReportEngine 报告流水线（HTML/MD/PDF）
└── core/
    ├── llm.py           # LLM 抽象（LangChain ChatOpenAI / MockLLM 降级）
    ├── cache.py         # Redis 缓存（宕机自动降级内存）
    ├── database.py      # SQLAlchemy 异步双数据源
    ├── task_manager.py  # 任务队列 + SSE 进度 + 断线重连 + 历史恢复
    └── observability.py # LangSmith 追踪 + token 成本统计
scripts/
├── build_eval_set.py    # 生成 60 条评测集
├── evaluate.py          # 评测流水线（完成率 / 工具成功率 / 成本）
├── bench.py             # 并发压测 + 串并行 P95 对比
└── bench_fault.py       # 故障注入对比评测
tests/                   # pytest（16 个用例全通过）
```

## 生产化说明

- **真实 LLM**：`LLM_API_KEY` 填 DeepSeek / 任意 OpenAI 兼容 Key，经 LangChain ChatOpenAI 自动切换真实模型（bind_tools 驱动 Function Calling）；留空自动使用 MockLLM；
- **真实数据源**：`app/agents/tools.py` 中演示数据源可替换为 MindSpider 抓取 + MySQL 落库；
- **数据库迁移**：开发环境 `create_all` 建表，生产建议 Alembic；
- **可观测**：配置 `LANGSMITH_API_KEY` 后自动上报链路追踪。

## 许可

MIT
