# NMS Agent

面向内部运维人员的对话式智能体（Agent）。基于 LangGraph 编排，FastAPI 提供 Web 对话与 SSE 流式接口；工具调用统一通过 **MCP 网关** 接入后端中台（NMS），并已落地长期记忆、上下文压缩、人工确认、语音输入/播报与会话持久化。

> 当前能力：对话 + Supervisor 路由 + IoT 设备/指标/拓扑/告警查询（经 MCP 网关）+ Skill 注入 + 流式返回 + 审批闸门 + 确定性 Report + PostgreSQL 会话/消息/摘要/长期记忆持久化 + LangGraph checkpoint 中断恢复 + StepFun 语音转写与播报。

## 技术栈

- **后端**：Python 3.12 / FastAPI / LangGraph / LangChain
- **模型**：OpenAI 兼容接口（当前可配置智谱 GLM / DeepSeek；接入 DeepSeek 等供应商需 `LLM_DISABLE_THINKING=true`，见 `config/.env`）
- **工具接入**：MCP 协议。独立进程 `mcp_gateway/`（FastMCP）读取 `config/gateways.yaml`，把中台 REST API 与外部 MCP Server 统一暴露为工具；Agent 侧作为 MCP 客户端消费
- **数据库**：PostgreSQL（`aiagent` schema，`aiagent_*` 业务表）；pgvector 语义检索（`VECTOR(768)`，HNSW 余弦）与 LangGraph checkpoint 均已接入运行流程
- **数据访问**：`psycopg_pool` 同步连接池（经线程池调用）+ 仓储层 `app/db/repositories/`
- **向量化**：本地 `sentence-transformers`（默认 `BAAI/bge-base-zh-v1.5`，768 维）
- **配置**：Pydantic Settings + `.env` + `config/agents.yaml` + `config/gateways.yaml`
- **可观测性**：结构化 JSON 日志（`trace_id`）+ 可选 Phoenix / OpenTelemetry 分布式追踪

## 环境要求

- Python >= 3.11（推荐 3.12）
- PostgreSQL >= 15；需安装 `vector` 扩展（pgvector 0.5+）
- 一个可用的 LLM API Key（OpenAI 兼容）
- MCP 网关独立进程（默认 `127.0.0.1:8001`），Agent 通过 `MCP_GATEWAY_URL` / `MCP_GATEWAY_TOKEN` 连接

## 项目结构

```
app/             # Agent 应用（编排、API、记忆、上下文、持久化）
mcp_gateway/     # 独立的 MCP 工具网关进程（配置驱动的 REST/联邦工具）
config/          # .env、agents.yaml（Agent 清单）与 gateways.yaml（工具网关）
prompts/         # 与代码分离的 Prompt（base/supervisor/conversation/iot/report/记忆/摘要）
skills/          # 领域 Skill 文档（SKILL.md，按 manifest 注入系统 Prompt）
database/sql/    # PostgreSQL 业务表基线 SQL（运行时以 aiagent_database_v1.1.sql 为准）
frontend/        # 静态对话前端（SSE 消费、审批弹窗、语音录制/播报）
doc/             # 产品与模块文档
tests/           # 测试用例
```

## 开发约定

- 每次有意义的改动先提交并推送到远程，避免互相覆盖（见下方 Git 备份）。
- 启动时必须完成全部配置加载与校验；配置缺失或工具 schema 不完整应阻止启动。
- 虚拟环境与 `.env` 必须加入 `.gitignore`，不得提交。

## Git 备份与恢复

日常提交：

```bash
git add -A
git commit -m "描述本次改动"
git push origin main
```

代码被覆盖时恢复昨天版本：

```bash
git checkout -- .                 # 工作区未提交，回退到最近一次提交
git reset --hard <昨天的commit>   # 已提交但未推送，回退到指定提交
```

只补回丢失的部分（不整体回滚）：

```bash
git checkout -b recover <昨天的commit>   # 基于昨天新建分支
# 手动把今天正确的新改动复制回来，或 git cherry-pick <今天的commit>
git checkout main && git merge recover   # 确认无误后合并
```

## 许可证

内部项目，仅供内部运维使用。
