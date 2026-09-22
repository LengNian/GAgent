-- 使用 psql 执行：
-- psql "$DATABASE_URL" -f database/sql/import_long_term_memories_from_tsv.sql
--
-- 前置条件：aiagent_threads 中已恢复本文件 source_thread_id 对应的会话。
-- 本脚本故意保留原始 memory_id，以维持 supersedes_memory_id 的替代关系。

BEGIN;
SET LOCAL TIME ZONE 'Asia/Shanghai';

CREATE TEMP TABLE import_long_term_memories (
    memory_id TEXT,
    user_id TEXT,
    content TEXT,
    memory_type TEXT,
    subject TEXT,
    attribute TEXT,
    importance TEXT,
    source_thread_id TEXT,
    supersedes_memory_id TEXT,
    is_active TEXT,
    last_used_at TEXT,
    inactive_reason TEXT,
    inactivated_at TEXT,
    created_at TEXT,
    updated_at TEXT
) ON COMMIT DROP;

\copy import_long_term_memories FROM '/home/user/G-Agent/aiagent_long_term_memories.txt' WITH (FORMAT csv, HEADER true, DELIMITER E'\t', QUOTE '"');

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM import_long_term_memories AS source
        LEFT JOIN aiagent.aiagent_threads AS thread
          ON thread.thread_id = NULLIF(source.source_thread_id, '')::UUID
         AND thread.user_id = source.user_id
        WHERE NULLIF(source.source_thread_id, '') IS NOT NULL
          AND thread.thread_id IS NULL
    ) THEN
        RAISE EXCEPTION
            'Missing source threads. Restore aiagent_threads before importing long-term memories.';
    END IF;
END $$;

INSERT INTO aiagent.aiagent_long_term_memories (
    memory_id, user_id, content, memory_type, subject, attribute, importance,
    source_thread_id, supersedes_memory_id, is_active, last_used_at,
    inactive_reason, inactivated_at, created_at, updated_at
)
OVERRIDING SYSTEM VALUE
SELECT
    memory_id::BIGINT,
    user_id,
    content,
    memory_type,
    subject,
    attribute,
    importance::SMALLINT,
    NULLIF(source_thread_id, '')::UUID,
    NULLIF(supersedes_memory_id, '')::BIGINT,
    is_active::BOOLEAN,
    to_timestamp(substring(last_used_at FROM '^\\d{1,2}/\\d{1,2}/\\d{4} \\d{2}:\\d{2}:\\d{2}'), 'DD/MM/YYYY HH24:MI:SS'),
    NULLIF(inactive_reason, ''),
    to_timestamp(substring(inactivated_at FROM '^\\d{1,2}/\\d{1,2}/\\d{4} \\d{2}:\\d{2}:\\d{2}'), 'DD/MM/YYYY HH24:MI:SS'),
    to_timestamp(substring(created_at FROM '^\\d{1,2}/\\d{1,2}/\\d{4} \\d{2}:\\d{2}:\\d{2}'), 'DD/MM/YYYY HH24:MI:SS'),
    to_timestamp(substring(updated_at FROM '^\\d{1,2}/\\d{1,2}/\\d{4} \\d{2}:\\d{2}:\\d{2}'), 'DD/MM/YYYY HH24:MI:SS')
FROM import_long_term_memories;

SELECT setval(
    pg_get_serial_sequence('aiagent.aiagent_long_term_memories', 'memory_id'),
    COALESCE(MAX(memory_id), 1),
    MAX(memory_id) IS NOT NULL
)
FROM aiagent.aiagent_long_term_memories;

COMMIT;
