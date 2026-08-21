# Database SQL

## Files

- `combined.sql`: one PostgreSQL database containing both relational tables and pgvector tables.
- `separate/relational.sql`: relational database instance tables.
- `separate/vector.sql`: independent pgvector database tables.

## Execution

Run each file while connected to its target database. The scripts create the `aiagent` schema
and use `VECTOR(768)`. They are design-baseline DDL; production changes should be managed by
Alembic migrations.

The separate deployment intentionally has no foreign keys from the vector database to the
relational database. `memory_id`, `chunk_id`, `document_id`, and `knowledge_base_id` are logical
IDs maintained by the application, and vector writes must be idempotent and retryable.
