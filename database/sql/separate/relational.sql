-- Separate deployment: relational database instance.
-- Run this file in nms_biz.

CREATE SCHEMA IF NOT EXISTS aiagent;
SET search_path TO aiagent, public;

CREATE TABLE aiagent.threads (
  thread_id        UUID PRIMARY KEY,
  user_id          TEXT NOT NULL CHECK (btrim(user_id) <> ''),
  title            TEXT,
  title_is_custom  BOOLEAN NOT NULL DEFAULT FALSE,
  next_message_seq BIGINT NOT NULL DEFAULT 0 CHECK (next_message_seq >= 0),
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (thread_id, user_id)
);

CREATE INDEX idx_threads_user_updated
  ON aiagent.threads (user_id, updated_at DESC);

CREATE TABLE aiagent.messages (
  thread_id  UUID NOT NULL REFERENCES aiagent.threads(thread_id) ON DELETE CASCADE,
  seq        BIGINT NOT NULL CHECK (seq > 0),
  role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
  content    TEXT NOT NULL CHECK (btrim(content) <> ''),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (thread_id, seq)
);

CREATE TABLE aiagent.long_term_memories (
  memory_id        BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  user_id          TEXT NOT NULL CHECK (btrim(user_id) <> ''),
  content          TEXT NOT NULL CHECK (btrim(content) <> ''),
  source_thread_id UUID,
  is_active        BOOLEAN NOT NULL DEFAULT TRUE,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  FOREIGN KEY (source_thread_id, user_id)
    REFERENCES aiagent.threads(thread_id, user_id)
    ON DELETE SET NULL (source_thread_id)
);

CREATE INDEX idx_long_term_memories_active_user
  ON aiagent.long_term_memories (user_id, created_at DESC)
  WHERE is_active;

CREATE TABLE aiagent.thread_summaries (
  thread_id       UUID PRIMARY KEY
                  REFERENCES aiagent.threads(thread_id) ON DELETE CASCADE,
  summary         TEXT NOT NULL CHECK (btrim(summary) <> ''),
  covered_to_seq  BIGINT NOT NULL CHECK (covered_to_seq > 0),
  summary_version INT NOT NULL DEFAULT 1 CHECK (summary_version > 0),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  FOREIGN KEY (thread_id, covered_to_seq)
    REFERENCES aiagent.messages(thread_id, seq) ON DELETE CASCADE
);

CREATE TABLE aiagent.knowledge_bases (
  knowledge_base_id UUID PRIMARY KEY,
  name              TEXT NOT NULL CHECK (btrim(name) <> ''),
  description       TEXT,
  scope_type        TEXT NOT NULL CHECK (scope_type IN ('global', 'user')),
  scope_id          TEXT,
  is_active         BOOLEAN NOT NULL DEFAULT TRUE,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (scope_type = 'global' OR btrim(scope_id) <> '')
);

CREATE INDEX idx_knowledge_bases_scope_active
  ON aiagent.knowledge_bases (scope_type, scope_id)
  WHERE is_active;

CREATE TABLE aiagent.knowledge_documents (
  document_id       UUID PRIMARY KEY,
  knowledge_base_id UUID NOT NULL
                    REFERENCES aiagent.knowledge_bases(knowledge_base_id)
                    ON DELETE CASCADE,
  title             TEXT NOT NULL CHECK (btrim(title) <> ''),
  source_uri        TEXT NOT NULL CHECK (btrim(source_uri) <> ''),
  content_type      TEXT NOT NULL CHECK (btrim(content_type) <> ''),
  checksum          TEXT NOT NULL CHECK (btrim(checksum) <> ''),
  version           INT NOT NULL DEFAULT 1 CHECK (version > 0),
  status            TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'failed')),
  metadata          JSONB,
  created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (knowledge_base_id, checksum, version)
);

CREATE INDEX idx_knowledge_documents_ready
  ON aiagent.knowledge_documents (knowledge_base_id, updated_at DESC)
  WHERE status = 'ready';

CREATE TABLE aiagent.knowledge_chunks (
  chunk_id    BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  document_id UUID NOT NULL
             REFERENCES aiagent.knowledge_documents(document_id)
             ON DELETE CASCADE,
  chunk_index INT NOT NULL CHECK (chunk_index >= 0),
  content     TEXT NOT NULL CHECK (btrim(content) <> ''),
  token_count INT CHECK (token_count IS NULL OR token_count > 0),
  metadata    JSONB,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_id, chunk_index)
);
