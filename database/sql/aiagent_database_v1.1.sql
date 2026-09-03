-- =============================================================================
-- aiagent_*  Agent 业务建表脚本（v1.0）
--
-- 设计基准  : 3-document/aiagent_database.md（由 1-input/agent_database.md 与 agent_database.sql 抽取整理）
-- 关联文档  : 1-input/agent_database.md（数据库部署与运维说明）
-- 目标数据库: PostgreSQL 15+（需安装 pgvector 扩展，VECTOR/HNSW；`ON DELETE SET NULL (col)` 部分列置空为 PG15+ 特性）
--
-- v1.0 设计要点：
--   1. 表名统一加 `aiagent_` 前缀（源表 threads / messages / long_term_memories /
--      semantic_memories / thread_summaries / knowledge_bases / knowledge_documents /
--      knowledge_chunks / knowledge_chunk_embeddings），业务表统一位于 `aiagent` schema，
--      `public` schema 仅用于 PostgreSQL 扩展对象（pgvector）；
--   2. 保留源设计完整关系：会话-消息复合主键、长期记忆-会话复合外键（部分列 SET NULL）、
--      会话摘要-消息复合外键、知识库四级层级（基础库-文档-块-向量）级联删除；
--   3. 统一命名：约束名 pk_/uk_/fk_/chk_ 前缀 + 表名 + 关键列，索引名 idx_ + 表名 + 用途，
--      HNSW 向量索引用于余弦相似度检索（pgvector 0.5+）；
--   4. 内容/标识非空校验统一使用 CHECK (btrim(col) <> '')（同源 SQL）；
--   5. LangGraph checkpoint 表（checkpoints / checkpoint_writes / checkpoint_blobs）
--      由 PostgresSaver.setup() 自动创建维护，不在本脚本范围内，禁止手写 DDL。
--
-- 使用提示：
--   - 整体包裹事务且 DROP 按外键依赖逆序，可重复执行；建议仅用于全新库初始化，
--   - 建表前需安装 pgvector 扩展（CREATE EXTENSION vector），向量维度固定 768，
--     不得直接修改现有列维度（须与 Embedding 模型配置一致）。
-- =============================================================================

BEGIN;

-- -----------------------------------------------------------------------------
-- 0. 扩展与 schema（幂等，可重复执行）
-- -----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE SCHEMA IF NOT EXISTS aiagent;
SET search_path TO aiagent, public;

-- -----------------------------------------------------------------------------
-- 0.0 公共触发器函数（幂等，可重复执行）
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.fn_update_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- 0.1 清理旧对象（按外键依赖逆序，子表先删）
-- -----------------------------------------------------------------------------
DROP TABLE IF EXISTS aiagent.aiagent_knowledge_chunk_embeddings;
DROP TABLE IF EXISTS aiagent.aiagent_knowledge_chunks;
DROP TABLE IF EXISTS aiagent.aiagent_knowledge_documents;
DROP TABLE IF EXISTS aiagent.aiagent_knowledge_bases;
DROP TABLE IF EXISTS aiagent.aiagent_thread_summaries;
DROP TABLE IF EXISTS aiagent.aiagent_semantic_memories;
DROP TABLE IF EXISTS aiagent.aiagent_long_term_memories;
DROP TABLE IF EXISTS aiagent.aiagent_messages;
DROP TABLE IF EXISTS aiagent.aiagent_threads;


-- =============================================================================
-- 1. 会话与消息模块（Conversation & Message）
-- =============================================================================

-- 会话表：thread_id 为全局 UUID 会话主键，也是 LangGraph 使用的会话标识
CREATE TABLE aiagent.aiagent_threads (
    thread_id        UUID        NOT NULL,
    user_id          TEXT        NOT NULL,
    title            TEXT,
    title_is_custom  BOOLEAN     NOT NULL DEFAULT FALSE,
    next_message_seq BIGINT      NOT NULL DEFAULT 0,        -- 分配下一条消息序号时使用
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_threads PRIMARY KEY (thread_id),
    CONSTRAINT uk_aiagent_threads_thread_user UNIQUE (thread_id, user_id),
    CONSTRAINT chk_aiagent_threads_user_id CHECK (btrim(user_id) <> ''),
    CONSTRAINT chk_aiagent_threads_title CHECK (title IS NULL OR btrim(title) <> ''),
    CONSTRAINT chk_aiagent_threads_next_seq CHECK (next_message_seq >= 0)
);

CREATE INDEX idx_aiagent_threads_user_updated ON aiagent.aiagent_threads (user_id, updated_at DESC);

COMMENT ON TABLE aiagent.aiagent_threads IS '会话表：保存会话基本信息，thread_id 为全局 UUID 主键，也是 LangGraph 使用的会话标识';
COMMENT ON COLUMN aiagent.aiagent_threads.user_id IS '会话所属用户 ID';
COMMENT ON COLUMN aiagent.aiagent_threads.title IS '会话标题；为空时由应用根据首条用户消息生成';
COMMENT ON COLUMN aiagent.aiagent_threads.title_is_custom IS '是否为用户手动设置的标题';
COMMENT ON COLUMN aiagent.aiagent_threads.next_message_seq IS '分配下一条消息序号时使用，必须 >= 0';

-- 消息表：复合主键 (thread_id, seq)，role 仅允许 user/assistant，会话删除时级联删除
CREATE TABLE aiagent.aiagent_messages (
    thread_id  UUID        NOT NULL,
    seq        BIGINT      NOT NULL,
    role       TEXT        NOT NULL,                        -- user | assistant
    content    TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_messages PRIMARY KEY (thread_id, seq),
    CONSTRAINT fk_aiagent_messages_thread FOREIGN KEY (thread_id) REFERENCES aiagent.aiagent_threads (thread_id) ON DELETE CASCADE,
    CONSTRAINT chk_aiagent_messages_seq CHECK (seq > 0),
    CONSTRAINT chk_aiagent_messages_role CHECK (role IN ('user', 'assistant')),
    CONSTRAINT chk_aiagent_messages_content CHECK (btrim(content) <> '')
);

COMMENT ON TABLE aiagent.aiagent_messages IS '消息表：保存页面需要展示的用户和助手文本，复合主键 (thread_id, seq)，会话删除时消息级联删除';
COMMENT ON COLUMN aiagent.aiagent_messages.role IS '消息发送方角色，仅允许 user / assistant';
COMMENT ON COLUMN aiagent.aiagent_messages.content IS '页面展示的消息正文，不能为空白';


-- =============================================================================
-- 2. 长期记忆模块（Long-term Memory）
-- =============================================================================

-- 长期记忆表：用户明确要求记住的事实文本；停用使用 is_active=FALSE，不建议物理删除
CREATE TABLE aiagent.aiagent_long_term_memories (
    memory_id        BIGINT      NOT NULL GENERATED ALWAYS AS IDENTITY,
    user_id          TEXT        NOT NULL,
    content          TEXT        NOT NULL,
    source_thread_id UUID,
    is_active        BOOLEAN     NOT NULL DEFAULT TRUE,     -- FALSE=不参与记忆检索
    created_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_long_term_memories PRIMARY KEY (memory_id),
    CONSTRAINT fk_aiagent_ltm_source_thread FOREIGN KEY (source_thread_id, user_id)
        REFERENCES aiagent.aiagent_threads (thread_id, user_id) ON DELETE SET NULL (source_thread_id),
    CONSTRAINT chk_aiagent_ltm_user_id CHECK (btrim(user_id) <> ''),
    CONSTRAINT chk_aiagent_ltm_content CHECK (btrim(content) <> '')
);

-- 仅索引启用中的记忆（按用户最近创建排序）
CREATE INDEX idx_aiagent_ltm_active_user ON aiagent.aiagent_long_term_memories (user_id, created_at DESC) WHERE is_active;

COMMENT ON TABLE aiagent.aiagent_long_term_memories IS '长期记忆表：保存用户明确要求记住的事实文本，通过 user_id 归属用户；停用记忆用 is_active=FALSE，不建议物理删除作为常规失效方式';
COMMENT ON COLUMN aiagent.aiagent_long_term_memories.source_thread_id IS '产生记忆的来源会话，复合外键 (source_thread_id, user_id) 关联会话表，删除来源会话时仅将 source_thread_id 置空';

-- 语义记忆向量表：保存长期记忆的向量索引，不保存原始事实文本
CREATE TABLE aiagent.aiagent_semantic_memories (
    semantic_memory_id BIGINT      NOT NULL GENERATED ALWAYS AS IDENTITY,
    memory_id          BIGINT      NOT NULL,
    embedding_model    TEXT        NOT NULL,
    embedding          VECTOR(768) NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_semantic_memories PRIMARY KEY (semantic_memory_id),
    CONSTRAINT fk_aiagent_sem_mem_memory FOREIGN KEY (memory_id)
        REFERENCES aiagent.aiagent_long_term_memories (memory_id) ON DELETE CASCADE,
    CONSTRAINT uk_aiagent_sem_mem_memory_model UNIQUE (memory_id, embedding_model),
    CONSTRAINT chk_aiagent_sem_mem_model CHECK (btrim(embedding_model) <> '')
);

CREATE INDEX idx_aiagent_sem_mem_model ON aiagent.aiagent_semantic_memories (embedding_model);
-- HNSW 余弦相似度索引，用于向量检索（需 pgvector 0.5+）
CREATE INDEX idx_aiagent_sem_mem_embedding ON aiagent.aiagent_semantic_memories USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE aiagent.aiagent_semantic_memories IS '语义记忆向量表：保存长期记忆的向量索引，不保存原始事实文本；同一 memory_id 在同一 embedding_model 下只能有一条索引，向量维度固定 VECTOR(768)';


-- =============================================================================
-- 3. 会话摘要模块（Thread Summary）
-- =============================================================================

-- 会话摘要表：每个会话最多一条当前摘要，covered_to_seq 经复合外键保证消息存在
CREATE TABLE aiagent.aiagent_thread_summaries (
    thread_id       UUID        NOT NULL,
    summary         TEXT        NOT NULL,
    covered_to_seq  BIGINT      NOT NULL,
    summary_version INT         NOT NULL DEFAULT 1,         -- 摘要版本号
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_thread_summaries PRIMARY KEY (thread_id),
    CONSTRAINT fk_aiagent_thread_summaries_thread FOREIGN KEY (thread_id)
        REFERENCES aiagent.aiagent_threads (thread_id) ON DELETE CASCADE,
    CONSTRAINT fk_aiagent_thread_summaries_covered_msg FOREIGN KEY (thread_id, covered_to_seq)
        REFERENCES aiagent.aiagent_messages (thread_id, seq) ON DELETE CASCADE,
    CONSTRAINT chk_aiagent_thread_summaries_summary CHECK (btrim(summary) <> ''),
    CONSTRAINT chk_aiagent_thread_summaries_covered_seq CHECK (covered_to_seq > 0),
    CONSTRAINT chk_aiagent_thread_summaries_version CHECK (summary_version > 0)
);

COMMENT ON TABLE aiagent.aiagent_thread_summaries IS '会话摘要表：每个会话最多一条当前摘要，covered_to_seq 表示摘要覆盖到的最后一条业务消息序号，并通过复合外键保证该消息存在';
COMMENT ON COLUMN aiagent.aiagent_thread_summaries.summary IS '压缩后的历史对话摘要，不能为空白';


-- =============================================================================
-- 4. 知识库模块（Knowledge Base：基础库-文档-块-向量 四级层级）
-- =============================================================================

-- 知识库表：知识库和访问范围，scope_type=global 时 scope_id 可空，否则非空
CREATE TABLE aiagent.aiagent_knowledge_bases (
    knowledge_base_id UUID        NOT NULL,
    name              TEXT        NOT NULL,
    description       TEXT,
    scope_type        TEXT        NOT NULL,                 -- global | user
    scope_id          TEXT,                                 -- 用户/团队范围标识
    is_active         BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_knowledge_bases PRIMARY KEY (knowledge_base_id),
    CONSTRAINT chk_aiagent_kb_name CHECK (btrim(name) <> ''),
    CONSTRAINT chk_aiagent_kb_scope_type CHECK (scope_type IN ('global', 'user')),
    CONSTRAINT chk_aiagent_kb_scope_id CHECK (scope_type = 'global' OR btrim(scope_id) <> '')
);

-- 仅索引启用的知识库（按访问范围筛选）
CREATE INDEX idx_aiagent_kb_scope_active ON aiagent.aiagent_knowledge_bases (scope_type, scope_id) WHERE is_active;

COMMENT ON TABLE aiagent.aiagent_knowledge_bases IS '知识库表：知识库和访问范围，支持 global/user 两类访问范围；scope_type=global 时 scope_id 可空，否则非空';
COMMENT ON COLUMN aiagent.aiagent_knowledge_bases.scope_type IS '知识库访问范围类型，仅允许 global / user';
COMMENT ON COLUMN aiagent.aiagent_knowledge_bases.scope_id IS '用户或团队范围标识，global 时可空，否则非空';

-- 知识库文档表：文档元数据、校验值、版本和处理状态；知识库删除时级联删除文档
CREATE TABLE aiagent.aiagent_knowledge_documents (
    document_id       UUID        NOT NULL,
    knowledge_base_id UUID        NOT NULL,
    title             TEXT        NOT NULL,
    source_uri        TEXT        NOT NULL,
    content_type      TEXT        NOT NULL,
    checksum          TEXT        NOT NULL,
    version           INT         NOT NULL DEFAULT 1,
    status            TEXT        NOT NULL,                 -- pending | ready | failed
    metadata          JSONB,                                -- 页码/来源等扩展元数据
    created_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_knowledge_documents PRIMARY KEY (document_id),
    CONSTRAINT fk_aiagent_kd_base FOREIGN KEY (knowledge_base_id)
        REFERENCES aiagent.aiagent_knowledge_bases (knowledge_base_id) ON DELETE CASCADE,
    CONSTRAINT uk_aiagent_kd_base_checksum_version UNIQUE (knowledge_base_id, checksum, version),
    CONSTRAINT chk_aiagent_kd_title CHECK (btrim(title) <> ''),
    CONSTRAINT chk_aiagent_kd_source_uri CHECK (btrim(source_uri) <> ''),
    CONSTRAINT chk_aiagent_kd_content_type CHECK (btrim(content_type) <> ''),
    CONSTRAINT chk_aiagent_kd_checksum CHECK (btrim(checksum) <> ''),
    CONSTRAINT chk_aiagent_kd_version CHECK (version > 0),
    CONSTRAINT chk_aiagent_kd_status CHECK (status IN ('pending', 'ready', 'failed'))
);

-- 仅索引处理完成的文档（按知识库分组、最近更新排序）
CREATE INDEX idx_aiagent_kd_ready ON aiagent.aiagent_knowledge_documents (knowledge_base_id, updated_at DESC) WHERE status = 'ready';

COMMENT ON TABLE aiagent.aiagent_knowledge_documents IS '知识库文档表：文档元数据、校验值、版本和处理状态；同一知识库中相同内容（checksum）的同一版本不能重复，知识库删除时级联删除文档';
COMMENT ON COLUMN aiagent.aiagent_knowledge_documents.checksum IS '文档内容校验值，不能为空白';
COMMENT ON COLUMN aiagent.aiagent_knowledge_documents.status IS '文档处理状态，仅允许 pending / ready / failed';

-- 知识库文档块表：文档切分后的文本块；同一文档的 chunk_index 唯一，文档删除时级联删除块
CREATE TABLE aiagent.aiagent_knowledge_chunks (
    chunk_id    BIGINT      NOT NULL GENERATED ALWAYS AS IDENTITY,
    document_id UUID        NOT NULL,
    chunk_index INT         NOT NULL,
    content     TEXT        NOT NULL,
    token_count INT,                                        -- token 数量，NULL 或 > 0
    metadata    JSONB,                                      -- 页码/章节等文档块元数据
    created_at  TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_knowledge_chunks PRIMARY KEY (chunk_id),
    CONSTRAINT fk_aiagent_kc_document FOREIGN KEY (document_id)
        REFERENCES aiagent.aiagent_knowledge_documents (document_id) ON DELETE CASCADE,
    CONSTRAINT uk_aiagent_kc_document_index UNIQUE (document_id, chunk_index),
    CONSTRAINT chk_aiagent_kc_chunk_index CHECK (chunk_index >= 0),
    CONSTRAINT chk_aiagent_kc_content CHECK (btrim(content) <> ''),
    CONSTRAINT chk_aiagent_kc_token_count CHECK (token_count IS NULL OR token_count > 0)
);

COMMENT ON TABLE aiagent.aiagent_knowledge_chunks IS '知识库文档块表：文档切分后的文本块；同一文档的 chunk_index 唯一，文档删除时级联删除块';
COMMENT ON COLUMN aiagent.aiagent_knowledge_chunks.chunk_index IS '文档块在原文中的顺序，必须 >= 0，同一文档内唯一';

-- 知识库文档块向量表：同一文档块在同一模型下只能有一条向量，文档块删除时级联删除向量索引
CREATE TABLE aiagent.aiagent_knowledge_chunk_embeddings (
    embedding_id    BIGINT      NOT NULL GENERATED ALWAYS AS IDENTITY,
    chunk_id        BIGINT      NOT NULL,
    embedding_model TEXT        NOT NULL,
    embedding       VECTOR(768) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_knowledge_chunk_embeddings PRIMARY KEY (embedding_id),
    CONSTRAINT fk_aiagent_kce_chunk FOREIGN KEY (chunk_id)
        REFERENCES aiagent.aiagent_knowledge_chunks (chunk_id) ON DELETE CASCADE,
    CONSTRAINT uk_aiagent_kce_chunk_model UNIQUE (chunk_id, embedding_model),
    CONSTRAINT chk_aiagent_kce_model CHECK (btrim(embedding_model) <> '')
);

CREATE INDEX idx_aiagent_kce_model ON aiagent.aiagent_knowledge_chunk_embeddings (embedding_model);
-- HNSW 余弦相似度索引，用于文档向量检索（需 pgvector 0.5+）
CREATE INDEX idx_aiagent_kce_embedding ON aiagent.aiagent_knowledge_chunk_embeddings USING hnsw (embedding vector_cosine_ops);

COMMENT ON TABLE aiagent.aiagent_knowledge_chunk_embeddings IS '知识库文档块向量表：文档块向量；同一文档块在同一模型下只能有一条向量，向量维度固定 VECTOR(768)，文档块删除时级联删除向量索引';


-- =============================================================================
-- 5. 触发器（BEFORE UPDATE 自动维护 updated_at）
-- -----------------------------------------------------------------------------
CREATE TRIGGER trg_aiagent_threads_modify BEFORE UPDATE ON aiagent.aiagent_threads
FOR EACH ROW EXECUTE FUNCTION public.fn_update_modified_at();
CREATE TRIGGER trg_aiagent_long_term_memories_modify BEFORE UPDATE ON aiagent.aiagent_long_term_memories
FOR EACH ROW EXECUTE FUNCTION public.fn_update_modified_at();
CREATE TRIGGER trg_aiagent_semantic_memories_modify BEFORE UPDATE ON aiagent.aiagent_semantic_memories
FOR EACH ROW EXECUTE FUNCTION public.fn_update_modified_at();
CREATE TRIGGER trg_aiagent_thread_summaries_modify BEFORE UPDATE ON aiagent.aiagent_thread_summaries
FOR EACH ROW EXECUTE FUNCTION public.fn_update_modified_at();
CREATE TRIGGER trg_aiagent_knowledge_bases_modify BEFORE UPDATE ON aiagent.aiagent_knowledge_bases
FOR EACH ROW EXECUTE FUNCTION public.fn_update_modified_at();
CREATE TRIGGER trg_aiagent_knowledge_documents_modify BEFORE UPDATE ON aiagent.aiagent_knowledge_documents
FOR EACH ROW EXECUTE FUNCTION public.fn_update_modified_at();
CREATE TRIGGER trg_aiagent_knowledge_chunk_embeddings_modify BEFORE UPDATE ON aiagent.aiagent_knowledge_chunk_embeddings
FOR EACH ROW EXECUTE FUNCTION public.fn_update_modified_at();

COMMIT;
