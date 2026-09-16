-- 为已有数据库增加长期记忆增量抽取游标。
-- 新字段从 0 开始，因此首次执行会按现有消息生成一次长期记忆；如需跳过历史数据，执行后手动设为 next_message_seq。

ALTER TABLE aiagent.aiagent_threads
    ADD COLUMN IF NOT EXISTS long_term_memory_processed_seq BIGINT NOT NULL DEFAULT 0;

ALTER TABLE aiagent.aiagent_threads
    DROP CONSTRAINT IF EXISTS chk_aiagent_threads_long_term_memory_processed_seq;

ALTER TABLE aiagent.aiagent_threads
    ADD CONSTRAINT chk_aiagent_threads_long_term_memory_processed_seq
    CHECK (long_term_memory_processed_seq >= 0);

COMMENT ON COLUMN aiagent.aiagent_threads.long_term_memory_processed_seq
    IS '长期记忆增量抽取已成功处理到的消息序号，必须 >= 0';
