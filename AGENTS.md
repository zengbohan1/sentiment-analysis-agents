# AGENTS.md

多智能体舆情分析系统（FastAPI / LangChain / SQLAlchemy / PostgreSQL / Redis / Docker），多智能体协作完成舆情采集、分析与报告生成，开源项目。

## Agent skills

### Issue tracker

工单以 markdown 文件存在 `.scratch/<feature>/` 下，规格在 `.scratch/<feature>/spec.md`，工单为 `issues/NN-<slug>.md`。See `docs/agents/issue-tracker.md`.

### Triage labels

五个标准角色标签：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。See `docs/agents/triage-labels.md`.

### Domain docs

Single-context：`CONTEXT.md` 在仓库根，ADR 在 `docs/adr/`。See `docs/agents/domain.md`.

## 协作约定

- **提交规范**：commit message 简短、专业，只描述技术变更本身；仓库内容（README、文档、commit、issue）仅限项目技术范畴，不得出现求职、招聘、简历等相关内容。
- **数据真实性**：README 与文档中所有性能指标、测试数字必须来自仓库内实际运行/测试的实测结果（如工具调用成功率、并发数、成本测算），禁止估算或编造。
- **讲解偏好**：写代码时同步讲解设计取舍，说明为什么这样设计。
