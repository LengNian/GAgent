# Database SQL

## Files

- `aiagent_database_v1.1.sql`: **当前运行时基线**。创建 `aiagent` schema 下全部 `aiagent_*` 业务表（会话、消息、任务状态、长期记忆、记忆证据、语义向量、会话摘要、知识库四级）。整体包裹事务、`DROP` 按外键逆序，可重复执行；建议仅用于**全新库初始化**（会删除并重建全部业务表）。
- `20260914_add_long_term_memory_cursor.sql`: 增量迁移，为已有库的 `aiagent_threads` 增加长期记忆抽取游标 `long_term_memory_processed_seq`（v1.1 基线已含该列）。
- `20260921_add_message_emotion.sql`: 增量迁移，为 `aiagent_messages` 增加 `emotion` 列及取值 CHECK（v1.1 基线已含）。
- `import_long_term_memories_from_tsv.sql`: 从 TSV 批量导入长期记忆的辅助脚本。
- `combined.sql`: **早期设计基线**（表名未加 `aiagent_` 前缀，如 `aiagent.threads`），仅作历史参考，不作为运行时脚本。

> 早期文档提到的 `separate/relational.sql` 与 `separate/vector.sql`（关系库/向量库分库部署）当前仓库中**不存在**；本项目采用关系表与 pgvector 向量表同库方案。

## Execution

```bash
# 全新库
psql "$DATABASE_URL" -f database/sql/aiagent_database_v1.1.sql
# 已有库补齐增量（全新库无需执行，v1.1 已包含这些列）
psql "$DATABASE_URL" -f database/sql/20260914_add_long_term_memory_cursor.sql
psql "$DATABASE_URL" -f database/sql/20260921_add_message_emotion.sql
```

脚本创建 `aiagent` schema，`vector` 扩展安装在 `public` schema，搜索路径为 `aiagent, public`；向量维度固定 `VECTOR(768)`，须与 `LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS` 一致。生产变更应逐步改由 Alembic 迁移管理。

## Runtime

当前运行时以 `aiagent_database_v1.1.sql` 为准：`app/db/` 使用 `psycopg_pool` 读写 `aiagent_threads`、`aiagent_messages`、`aiagent_thread_states`、`aiagent_thread_summaries`、`aiagent_long_term_memories`、`aiagent_long_term_memory_evidence`、`aiagent_semantic_memories`。知识库四级表（`aiagent_knowledge_*`）为设计基线，尚未接入检索运行流程。LangGraph checkpoint 表（`checkpoints` / `checkpoint_writes` / `checkpoint_blobs`）由 `AsyncPostgresSaver.setup()` 自动创建维护，不在业务脚本内，禁止手写 DDL。
