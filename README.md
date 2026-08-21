# Multi-agent Sentiment Analysis

An AI multi-agent system for collecting, analysing, and reporting on public sentiment. It coordinates parallel Query, Media, and Insight agents through a custom scheduler and exposes long-running task progress over SSE.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)
![Tool success](https://img.shields.io/badge/Tool%20success-100%25-4C9F70?style=flat-square)
![Concurrency](https://img.shields.io/badge/Concurrency-60%20tasks-4C9F70?style=flat-square)

## Highlights

- **Parallel analysis**: Query, Media, and Insight agents share an `AgentContext` and execute under the custom `ForumEngine` scheduler.
- **Tool calling**: retrieval, document parsing, database querying, and sentiment analysis tools are registered for function calling.
- **Long-running task control**: queued tasks, concurrency limits, progress events, reconnectable SSE streams, and Redis snapshots.
- **Reporting**: produces HTML, Markdown, and PDF reports.
- **Resilience**: retries at tool and agent level; uses `MockLLM` when no LLM API key is configured.
- **Observability**: evaluation scripts, optional LangSmith tracing, and token-cost accounting.

Measured results: 265 real-LLM tool calls completed successfully; 60 offline `MockLLM` tasks completed in the concurrency benchmark; parallel real-LLM scheduling reduced P95 from 66.2s to 30.3s; average real-LLM task cost was 0.054 CNY.

## Architecture

```text
POST /v1/tasks
  -> TaskManager: queue, semaphore, SSE progress, Redis snapshots
  -> ForumEngine: Query + Media + Insight agents
  -> ReportEngine: HTML, Markdown, or PDF
  -> status, events, and downloadable report endpoints
```

## Quick start

### 1. Create the environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env

# macOS / Linux
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

`LLM_API_KEY` is optional. Leave it empty to use the offline `MockLLM` path.

### 2. Start the service

Choose one mode:

```bash
# Full Docker deployment: API + PostgreSQL + Redis
docker compose up -d --build

# Local API development after starting PostgreSQL and Redis yourself
# Windows: .venv\Scripts\python -m uvicorn app.main:app --port 8000
# macOS / Linux: .venv/bin/python -m uvicorn app.main:app --port 8000
```

For an offline local run, set `DATABASE_URL=sqlite+aiosqlite:///./sentiment.db` in `.env`. Redis failures degrade to in-memory cache behavior.

### 3. Submit a task

```bash
curl -X POST http://localhost:8000/v1/tasks \\
  -H "Content-Type: application/json" \\
  -d '{"query":"分析新品手机「星云 X1」近期的口碑舆情"}'

curl -N http://localhost:8000/v1/tasks/<task_id>/events
curl -o report.html http://localhost:8000/v1/reports/<task_id>?fmt=html
```

## API

| Method | Path | Description |
| --- | --- | --- |
| POST | `/v1/tasks` | Submit a task and receive a `task_id`. |
| GET | `/v1/tasks/{id}` | Read task state and progress. |
| GET | `/v1/tasks/{id}/events` | Stream SSE task progress. |
| POST | `/v1/tasks/{id}/cancel` | Cancel a queued or active task. |
| GET | `/v1/stats` | Read task statistics. |
| GET | `/v1/reports/{id}?fmt=html|md|pdf` | Download a report. |
| GET | `/health` | Health check. |

## Evaluation and tests

```bash
python scripts/build_eval_set.py
python scripts/evaluate.py --limit 60 --parallel
python scripts/bench.py --concurrency 60 --tasks 60
pytest tests -q
```

At the measured revision, 16 tests passed; core-module coverage was 64%, with agent-layer coverage between 80% and 96%.

## Project structure

```text
app/
├── main.py
├── agents/       # Query, Media, Insight agents and tools
├── engines/      # ForumEngine and ReportEngine
└── core/         # LLM, cache, database, task manager, observability
scripts/          # dataset build, evaluation, and benchmarks
tests/            # pytest suite
docker-compose.yml
```

## Production notes

- Configure `LLM_API_KEY` with a DeepSeek or OpenAI-compatible provider for real function-calling runs.
- The tool data sources are demonstrations and can be replaced with production crawlers and databases.
- Development uses `create_all`; use Alembic or an equivalent migration tool for production schemas.
- Configure `LANGSMITH_API_KEY` to enable optional trace reporting.

## License

[MIT](LICENSE)
