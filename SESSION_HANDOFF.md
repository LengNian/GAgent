# G-Agent 会话交接记录

更新时间：2026-09-01  
当前分支：`main`  
当前 HEAD：`fe483f2`（会话重命名 bug）

本文用于下一次开发会话快速恢复上下文。本文记录的是当前代码事实，不把 PRD 中的目标能力当作已完成实现。

## 1. 本次会话范围

本次会话完成了以下工作：

1. 对当前代码、数据库 SQL 和 `doc/` 文档进行了静态核对。
2. 确认应用已经接入 PostgreSQL，但范围仅是会话和业务消息基础持久化。
3. 将 README、模块文档、运维说明、实现总结和 PRD 状态注记同步到当前实现。
4. 未修改 Agent 业务逻辑；保留了会话开始前工作区已有的代码改动。

## 2. 关键决策

### 数据库边界

- 当前数据库能力定义为：`aiagent.threads` 和 `aiagent.messages` 的 PostgreSQL 持久化。
- 运行时使用 `app/database.py` 中的 `psycopg_pool.ConnectionPool`；API 通过 `anyio.to_thread.run_sync` 调用同步数据库代码。
- 每次数据库操作在连接上下文中提交或回滚；应用关闭时由 `app/main.py` 调用 `close_pool()`。
- 消息写入先更新 `threads.next_message_seq`，再插入 `messages`，依靠数据库行更新锁保证会话内序号不重复。
- 业务消息只保存 `user`/`assistant` 文本，不保存完整 LangChain JSON、工具原始输出或 token 指标。
- `database/sql/*.sql` 是建表基线，不是迁移系统；当前没有 SQLAlchemy 或 Alembic。

### 设计与实现的区分

- `long_term_memories`、`semantic_memories`、`thread_summaries` 和知识库表目前只有 DDL/设计，未接入业务流程。
- LangGraph 图当前通过 `compile()` 创建，尚未接入 `PostgresSaver`、checkpoint 或中断恢复。
- PRD 仍保留完整 P0/P1 目标；`doc/module/PRD.md` 顶部已增加实现状态注记，明确目标不等于完成清单。

### 当前会话与身份行为

- 已有基础接口：
  - `POST /api/threads`
  - `GET /api/threads`
  - `PATCH /api/threads/{thread_id}`
  - `DELETE /api/threads/{thread_id}`
  - `GET /api/threads/{thread_id}/messages`
  - `POST /api/threads/{thread_id}/chat`
- 前端点击“新对话”先创建本地草稿，首次发送时才创建数据库会话；这与 PRD 中“点击后立即创建”存在已知差异。
- 前端刷新时加载服务端会话列表，但不会自动打开上一次活动会话，需用户从列表选择。
- 查询均带 `user_id` 归属条件，但真实 `CurrentUser`/JWT/SSO 尚未接入；联调请求体可能提供用户 ID，不能视为可信鉴权。
- `_active_threads` 只适用于单进程，同一会话在多实例间没有共享锁。

## 3. 当前已完成工作

### 运行时代码（本次核对确认）

- `app/database.py`：连接池、事务、会话 CRUD、标题更新、消息读取和追加。
- `app/api/threads.py`：数据库会话 API、SSE 聊天、标题/删除接口，以及数据库配置错误的部分 `503` 映射。
- `app/main.py`：应用生命周期结束时释放数据库连接池。
- `database/sql/combined.sql`：包含 `threads/messages` 及未来记忆、摘要、知识库表的同库基线。

### 文档同步

- README：改为说明 PostgreSQL 会话/消息已接入，pgvector 仍未接入运行流程。
- 数据库文档：补充 `psycopg_pool`、事务、标题字段、基线 SQL 和未完成能力。
- 会话文档：补充列表/标题/删除 API、服务重启恢复和前端实际行为。
- 记忆、配置、安全、异常文档：统一当前状态和剩余差距。
- `doc/module/PRD.md`：增加实现状态注记，补充 `title/title_is_custom` 字段和当前额外 API。
- 运维说明与 SQL README：说明 `DATABASE_URL`、业务表前置条件和当前运行时范围。

## 4. 未完成事项

### 优先处理

1. **身份来源**：接入可信 `CurrentUser`，禁止从请求体信任 `user_id`；统一修复创建、读取、聊天、重命名和删除接口的身份来源。
2. **数据库配置契约**：明确 `DATABASE_URL` 是会话 API 的运行条件；增加连接、schema 和关键表健康检查，并统一数据库异常到 `503` 的映射。
3. **迁移策略**：决定采用 Alembic，补充现有数据库的 migration，尤其是 `title`/`title_is_custom` 字段，避免只依赖不可重复执行的基线 SQL。
4. **真实集成测试**：在测试 PostgreSQL 上验证建表、事务回滚、消息序号并发、级联删除、标题更新和服务重启恢复。

### 后续能力

- LangGraph `PostgresSaver` checkpoint、`configurable.thread_id` 和中断恢复。
- 短期记忆窗口、超窗摘要和 token 预算控制。
- `save_memory`、长期记忆失效、Embedding、pgvector 语义检索及降级路径。
- RAG 知识库导入、切块、Embedding 和检索。
- 多实例会话锁、用户/IP/全局限流。
- 后端最大输入长度、工具输出截断和更完整的数据库错误分类。

## 5. 当前工作区文件

### 本次会话新增或同步的文件

- `SESSION_HANDOFF.md`（本文件，新建）
- `README.md`
- `config/.env.example`
- `database/sql/README.md`
- `运维部署说明.md`
- `doc/module/CONFIGURATION.md`
- `doc/module/CONTEXT_SUMMARY.md`
- `doc/module/DATABASE.md`
- `doc/module/ERRORS_AND_RETRIES.md`
- `doc/module/MEMORY_SYSTEM.md`
- `doc/module/PRD.md`
- `doc/module/SECURITY_AND_AUTHORIZATION.md`
- `doc/module/SESSION_MANAGEMENT.md`
- `doc/process/database_operations_guide.md`
- `doc/process/implementation_summary.md`

### 会话开始前已存在、此次保留的代码改动

- `app/api/threads.py`
- `app/database.py`
- `frontend/app.js`
- `frontend/index.html`

`doc/` 当前被 `.gitignore` 忽略，因此文档修改不会出现在普通 `git status` 输出中；文件内容仍已更新。工作区中的上述代码改动均未回滚或覆盖。

## 6. 验证结果

- `.venv/bin/python -m unittest discover -s tests -q`：26 项通过。
- `.venv/bin/python -m compileall -q app tests`：通过。
- `node --check frontend/app.js`：通过。
- `git diff --check`：通过。
- 未执行真实 PostgreSQL 集成测试：当前 shell 未配置 `DATABASE_URL`。

## 7. 下一次会话建议顺序

1. 先阅读本文件、`doc/module/CONTEXT_SUMMARY.md` 和 `doc/process/database_operations_guide.md`。
2. 执行 `git status --short --untracked-files=all`，不要回滚已有改动。
3. 明确下一步是“继续文档/配置修正”还是“实现数据库能力”；不要把 DDL、checkpoint 或记忆设计误判为运行时功能。
4. 若开始数据库实现，先在测试 PostgreSQL 配置 `DATABASE_URL` 并验证 `aiagent.threads/messages`，再添加集成测试和迁移。
5. 身份鉴权应优先于多用户数据隔离和长期记忆功能。
