# NMS Agent

面向内部运维人员的对话式智能体（Agent）。基于 LangGraph 编排，FastAPI 提供 Web 对话与 SSE 流式接口，PostgreSQL + pgvector 持久化会话与语义记忆。

> 当前为 P0 最小闭环阶段：可对话、可流式返回、可持久化会话与长期记忆。详细需求见 `doc/PRD.md`。

## 技术栈

- **后端**：Python 3.12 / FastAPI / LangGraph / LangChain
- **模型**：OpenAI 兼容接口（当前接入智谱 BigModel，见 `config/.env`）
- **数据库**：PostgreSQL + pgvector 扩展
- **ORM / 迁移**：SQLAlchemy (async) / Alembic
- **配置**：Pydantic Settings + `.env` + `tools.yaml`

## 环境要求

- Python >= 3.11（推荐 3.12）
- PostgreSQL >= 14，并启用 `vector` 扩展
- 一个可用的 LLM API Key（OpenAI 兼容）

## 项目结构

```
app/             # 应用代码（agent 编排、API、配置、数据访问）
config/          # .env 与 tools.yaml 等配置文件
migrations/      # Alembic 迁移脚本
doc/             # PRD 等产品文档
tests/           # 测试用例
scripts/         # 辅助脚本
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
