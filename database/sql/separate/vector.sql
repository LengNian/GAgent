-- Separate deployment: vector database instance.
-- Run this file in nms_vector.
-- No foreign keys reference tables in the relational database.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;

CREATE SCHEMA IF NOT EXISTS aiagent;
SET search_path TO aiagent, public;

CREATE TABLE aiagent.semantic_memories (
  semantic_memory_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  memory_id          BIGINT NOT NULL,
  user_id            TEXT NOT NULL CHECK (btrim(user_id) <> ''),
  embedding_model    TEXT NOT NULL CHECK (btrim(embedding_model) <> ''),
  embedding          VECTOR(768) NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (memory_id, embedding_model)
);

CREATE INDEX idx_semantic_memories_user_model
  ON aiagent.semantic_memories (user_id, embedding_model);

CREATE INDEX idx_semantic_memories_embedding
  ON aiagent.semantic_memories USING hnsw (embedding vector_cosine_ops);

CREATE TABLE aiagent.knowledge_chunk_embeddings (
  embedding_id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  chunk_id           BIGINT NOT NULL,
  document_id        UUID NOT NULL,
  knowledge_base_id  UUID NOT NULL,
  embedding_model    TEXT NOT NULL CHECK (btrim(embedding_model) <> ''),
  embedding          VECTOR(768) NOT NULL,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (chunk_id, embedding_model)
);

CREATE INDEX idx_knowledge_chunk_embeddings_scope_model
  ON aiagent.knowledge_chunk_embeddings (knowledge_base_id, embedding_model);

CREATE INDEX idx_knowledge_chunk_embeddings_embedding
  ON aiagent.knowledge_chunk_embeddings USING hnsw (embedding vector_cosine_ops);
