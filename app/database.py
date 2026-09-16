"""PostgreSQL 持久化操作。"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import math
from typing import Any
from uuid import UUID

from app.long_term_memory_policy import is_single_value_long_term_memory_attribute
from app.settings import get_settings

_pool: Any | None = None


@dataclass(frozen=True)
class StoredMessage:
    """带持久化序号的业务消息，供上下文和摘要服务内部使用。"""

    seq: int
    role: str
    content: str


@dataclass(frozen=True)
class ThreadSummary:
    """会话当前滚动摘要及其覆盖范围。"""

    summary: str
    covered_to_seq: int
    summary_version: int
    summary_token_count: int


@dataclass(frozen=True)
class ThreadTaskState:
    """会话中待确认 Action 的持久化状态。"""

    state: dict[str, Any]
    state_version: int


@dataclass(frozen=True)
class LongTermMemoryInput:
    """已通过服务层校验、可事务化写入的长期记忆候选。"""

    content: str
    memory_type: str
    subject: str
    attribute: str
    importance: int
    evidence_message_seqs: tuple[int, ...]


@dataclass(frozen=True)
class PersistedLongTermMemory:
    """本次事务中新建、可继续生成语义索引的长期记忆。"""

    memory_id: int
    content: str


@dataclass(frozen=True)
class LongTermMemoryRecallCandidate:
    """pgvector 粗查返回的有效长期记忆候选。"""

    memory_id: int
    content: str
    memory_type: str
    subject: str
    attribute: str
    importance: int
    effective_last_used_at: datetime
    cosine_distance: float


def _get_pool() -> Any:
    """获取进程级 PostgreSQL 连接池，首次使用时延迟创建。"""

    global _pool
    if _pool is not None:
        return _pool
    database_url = get_settings().database_url
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        from psycopg_pool import ConnectionPool
    except ImportError as error:
        raise RuntimeError("PostgreSQL pool driver is missing; install psycopg[pool]") from error

    _pool = ConnectionPool(database_url, min_size=1, max_size=10, open=True)
    return _pool


@contextmanager
def _connection() -> Iterator[Any]:
    """从进程级连接池借用连接，并在退出时提交或回滚事务。"""

    with _get_pool().connection() as connection:
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def close_pool() -> None:
    """关闭进程级连接池，供应用生命周期结束时调用。"""

    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def create_thread(thread_id: UUID, user_id: str) -> None:
    """持久化新会话。"""

    with _connection() as connection:
        connection.execute(
            "INSERT INTO aiagent.aiagent_threads (thread_id, user_id) VALUES (%s, %s)",
            (thread_id, user_id),
        )


def thread_exists_for_user(thread_id: UUID, user_id: str) -> bool:
    """判断会话是否存在且属于指定用户。"""

    with _connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM aiagent.aiagent_threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        ).fetchone()
    return row is not None


def load_messages(thread_id: UUID, user_id: str) -> list[tuple[str, str]] | None:
    """读取用户所属会话的消息；会话不存在或不属于用户时返回 None。"""

    stored_messages = load_stored_messages(thread_id, user_id)
    if stored_messages is None:
        return None
    return [(message.role, message.content) for message in stored_messages]


def load_stored_messages(thread_id: UUID, user_id: str) -> list[StoredMessage] | None:
    """读取带序号的业务消息，供摘要覆盖范围和上下文窗口计算。"""

    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT m.seq, m.role, m.content
            FROM aiagent.aiagent_messages AS m
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = m.thread_id
            WHERE m.thread_id = %s AND t.user_id = %s
            ORDER BY m.seq
            """,
            (thread_id, user_id),
        ).fetchall()
        thread = connection.execute(
            "SELECT 1 FROM aiagent.aiagent_threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        ).fetchone()
    if thread is None:
        return None
    return [StoredMessage(int(seq), str(role), str(content)) for seq, role, content in rows]


def load_recent_stored_messages(
    thread_id: UUID,
    user_id: str,
    limit: int,
) -> list[StoredMessage] | None:
    """读取当前会话最近的业务消息，并保持原始 seq 顺序。

    Args:
        thread_id: 会话标识。
        user_id: 会话所属用户，用于隔离查询范围。
        limit: 返回的最大消息数量，必须为正数。
    Returns:
        最近消息；会话不存在或不属于用户时返回 None。
    Raises:
        ValueError: limit 不是正数。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 校验 limit，避免数据库接收无意义的窗口大小。
    # 2. 按 seq 倒序取得最近消息，再在 SQL 外恢复正序，供抽取模型识别证据序号。
    # 3. 单独确认会话归属，区分空会话与无权/不存在的会话。
    # =========================================================================
    if limit < 1:
        raise ValueError("limit must be positive")

    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT seq, role, content
            FROM (
                SELECT m.seq, m.role, m.content
                FROM aiagent.aiagent_messages AS m
                JOIN aiagent.aiagent_threads AS t ON t.thread_id = m.thread_id
                WHERE m.thread_id = %s AND t.user_id = %s
                ORDER BY m.seq DESC
                LIMIT %s
            ) AS recent_messages
            ORDER BY seq
            """,
            (thread_id, user_id, limit),
        ).fetchall()
        thread = connection.execute(
            "SELECT 1 FROM aiagent.aiagent_threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        ).fetchone()
    if thread is None:
        return None
    return [StoredMessage(int(seq), str(role), str(content)) for seq, role, content in rows]


def load_incremental_long_term_memory_messages(
    thread_id: UUID, user_id: str, through_seq: int
) -> tuple[list[StoredMessage], int] | None:
    """读取线程长期记忆游标之后、指定回复序号之前的消息。"""

    if through_seq < 1:
        raise ValueError("through_seq must be positive")
    with _connection() as connection:
        thread = connection.execute(
            """
            SELECT long_term_memory_processed_seq
            FROM aiagent.aiagent_threads
            WHERE thread_id = %s AND user_id = %s
            """,
            (thread_id, user_id),
        ).fetchone()
        if thread is None:
            return None
        processed_seq = int(thread[0])
        rows = connection.execute(
            """
            SELECT seq, role, content
            FROM aiagent.aiagent_messages
            WHERE thread_id = %s AND seq > %s AND seq <= %s
            ORDER BY seq
            """,
            (thread_id, processed_seq, through_seq),
        ).fetchall()
    return ([StoredMessage(int(seq), str(role), str(content)) for seq, role, content in rows], processed_seq)


def advance_long_term_memory_cursor(thread_id: UUID, user_id: str, through_seq: int) -> bool:
    """在抽取成功后幂等推进线程的长期记忆处理游标。"""

    if through_seq < 0:
        raise ValueError("through_seq must be non-negative")
    with _connection() as connection:
        row = connection.execute(
            """
            UPDATE aiagent.aiagent_threads
            SET long_term_memory_processed_seq = GREATEST(long_term_memory_processed_seq, %s)
            WHERE thread_id = %s AND user_id = %s
            RETURNING long_term_memory_processed_seq
            """,
            (through_seq, thread_id, user_id),
        ).fetchone()
    return row is not None


def touch_long_term_memory_groups(user_id: str, groups: list[tuple[str, str, str]]) -> int:
    """刷新实际注入模型的长期记忆属性组最近使用时间。"""
    if not groups:
        return 0
    updated = 0
    with _connection() as connection:
        for memory_type, subject, attribute in groups:
            result = connection.execute(
                """
                UPDATE aiagent.aiagent_long_term_memories
                SET last_used_at = now()
                WHERE user_id = %s AND memory_type = %s AND subject = %s
                  AND attribute = %s AND is_active = TRUE
                """,
                (user_id, memory_type, subject, attribute),
            )
            updated += result.rowcount
    return updated


def merge_long_term_memories(
    thread_id: UUID,
    user_id: str,
    memories: list[LongTermMemoryInput],
    through_seq: int,
    *,
    memory_embeddings: list[list[float] | None] | None = None,
    embedding_model: str | None = None,
    semantic_dedup_enabled: bool = False,
    semantic_dedup_distance: float = 0.12,
) -> list[PersistedLongTermMemory]:
    """去重、替代并持久化已校验的长期记忆及其消息级证据。

    Args:
        thread_id: 本轮候选所属会话。
        user_id: 当前用户，用于所有读写隔离。
        memories: 服务层完成 schema 与证据角色校验后的候选列表。
        through_seq: 本次成功抽取覆盖到的 Assistant 消息序号。
    Returns:
        实际新增且已在同一事务中写入可用向量的长期记忆。
    Raises:
        ValueError: 会话归属、证据消息或消息序号校验失败。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 拒绝空候选；该分支由服务层单独推进游标，无需开启记忆写入事务。
    # 2. 校验会话和每条证据均属于当前用户，数据库层再次防御服务层遗漏。
    # 3. 活跃记忆正文完全相同时跳过；多值属性再通过同组向量距离过滤语义重复。
    # 4. 单值属性不做语义去重，始终保留“新值替代旧值”的事实更新语义。
    # 5. 同一事务写入原文、向量、证据和游标，避免并发任务看不到同批新向量。
    # 6. 任意候选或游标更新失败均回滚整个事务，避免留下半完成状态。
    # =========================================================================
    if not memories:
        raise ValueError("Long-term memory merge requires at least one memory")
    if through_seq < 1:
        raise ValueError("through_seq must be positive")
    if memory_embeddings is not None and len(memory_embeddings) != len(memories):
        raise ValueError("memory_embeddings must match memories")
    normalized_embedding_model = (embedding_model or "").strip()
    if semantic_dedup_enabled and not normalized_embedding_model:
        raise ValueError("Embedding model is required for semantic deduplication")
    if not 0 <= semantic_dedup_distance <= 2:
        raise ValueError("semantic_dedup_distance must be between 0 and 2")
    if memory_embeddings is not None:
        for embedding in memory_embeddings:
            if embedding is not None and (
                len(embedding) != 768
                or not all(
                    isinstance(value, (int, float)) and math.isfinite(value)
                    for value in embedding
                )
            ):
                raise ValueError("Memory embedding must contain 768 finite values")

    created_memories: list[PersistedLongTermMemory] = []
    with _connection() as connection:
        owner = connection.execute(
            "SELECT 1 FROM aiagent.aiagent_threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        ).fetchone()
        if owner is None:
            raise ValueError("Thread not found for long-term memory persistence")

        for memory_index, memory in enumerate(memories):
            embedding = memory_embeddings[memory_index] if memory_embeddings is not None else None
            evidence_rows = connection.execute(
                """
                SELECT m.seq, m.role
                FROM aiagent.aiagent_messages AS m
                JOIN aiagent.aiagent_threads AS t ON t.thread_id = m.thread_id
                WHERE m.thread_id = %s
                  AND t.user_id = %s
                  AND m.seq = ANY(%s)
                """,
                (thread_id, user_id, list(memory.evidence_message_seqs)),
            ).fetchall()
            evidence_by_seq = {int(seq): str(role) for seq, role in evidence_rows}
            expected_evidence = set(memory.evidence_message_seqs)
            if set(evidence_by_seq) != expected_evidence or "user" not in evidence_by_seq.values():
                raise ValueError("Long-term memory evidence is invalid")

            duplicate = connection.execute(
                """
                SELECT memory_id
                FROM aiagent.aiagent_long_term_memories
                WHERE user_id = %s AND is_active = TRUE AND content = %s
                LIMIT 1
                """,
                (user_id, memory.content),
            ).fetchone()
            if duplicate is not None:
                continue

            if (
                semantic_dedup_enabled
                and embedding is not None
                and not is_single_value_long_term_memory_attribute(memory.attribute)
            ):
                # 仅靠 FOR UPDATE 无法锁住“尚不存在的相似记忆”；按用户属性组串行化去重判断。
                connection.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                        hashtext(%s),
                        hashtext(%s)
                    )
                    """,
                    (
                        user_id,
                        f"{memory.memory_type}:{memory.subject}:{memory.attribute}",
                    ),
                )
                vector_literal = "[" + ",".join(format(float(value), ".12g") for value in embedding) + "]"
                semantic_duplicate = connection.execute(
                    """
                    SELECT m.memory_id
                    FROM aiagent.aiagent_semantic_memories AS s
                    JOIN aiagent.aiagent_long_term_memories AS m
                      ON m.memory_id = s.memory_id
                    WHERE m.user_id = %s
                      AND m.memory_type = %s
                      AND m.subject = %s
                      AND m.attribute = %s
                      AND m.is_active = TRUE
                      AND s.embedding_model = %s
                      AND (s.embedding <=> %s::vector) <= %s
                    ORDER BY s.embedding <=> %s::vector ASC
                    LIMIT 1
                    FOR UPDATE OF m
                    """,
                    (
                        user_id,
                        memory.memory_type,
                        memory.subject,
                        memory.attribute,
                        normalized_embedding_model,
                        vector_literal,
                        semantic_dedup_distance,
                        vector_literal,
                    ),
                ).fetchone()
                if semantic_duplicate is not None:
                    continue

            replaced_rows = []
            if is_single_value_long_term_memory_attribute(memory.attribute):
                replaced_rows = connection.execute(
                    """
                    SELECT memory_id
                    FROM aiagent.aiagent_long_term_memories
                    WHERE user_id = %s
                      AND subject = %s
                      AND attribute = %s
                      AND is_active = TRUE
                    FOR UPDATE
                    """,
                    (user_id, memory.subject, memory.attribute),
                ).fetchall()
            replaced_memory_ids = [int(row[0]) for row in replaced_rows]

            row = connection.execute(
                """
                INSERT INTO aiagent.aiagent_long_term_memories (
                    user_id, content, memory_type, subject, attribute, importance,
                    source_thread_id, supersedes_memory_id
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING memory_id
                """,
                (
                    user_id,
                    memory.content,
                    memory.memory_type,
                    memory.subject,
                    memory.attribute,
                    memory.importance,
                    thread_id,
                    replaced_memory_ids[0] if replaced_memory_ids else None,
                ),
            ).fetchone()
            if row is None:
                raise RuntimeError("Long-term memory insert did not return memory_id")
            memory_id = int(row[0])

            if embedding is not None and normalized_embedding_model:
                vector_literal = "[" + ",".join(format(float(value), ".12g") for value in embedding) + "]"
                connection.execute(
                    """
                    INSERT INTO aiagent.aiagent_semantic_memories (
                        memory_id, embedding_model, embedding
                    )
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (memory_id, embedding_model)
                    DO UPDATE SET embedding = EXCLUDED.embedding,
                                  updated_at = CURRENT_TIMESTAMP
                    """,
                    (memory_id, normalized_embedding_model, vector_literal),
                )

            if replaced_memory_ids:
                connection.execute(
                    """
                    UPDATE aiagent.aiagent_long_term_memories
                    SET is_active = FALSE,
                        inactive_reason = 'superseded',
                        inactivated_at = now()
                    WHERE memory_id = ANY(%s)
                    """,
                    (replaced_memory_ids,),
                )

            for message_seq in memory.evidence_message_seqs:
                connection.execute(
                    """
                    INSERT INTO aiagent.aiagent_long_term_memory_evidence (
                        memory_id, thread_id, message_seq
                    )
                    VALUES (%s, %s, %s)
                    """,
                    (memory_id, thread_id, message_seq),
                )
            created_memories.append(PersistedLongTermMemory(memory_id, memory.content))
        cursor = connection.execute(
            """
            UPDATE aiagent.aiagent_threads
            SET long_term_memory_processed_seq = GREATEST(long_term_memory_processed_seq, %s)
            WHERE thread_id = %s AND user_id = %s
            RETURNING long_term_memory_processed_seq
            """,
            (through_seq, thread_id, user_id),
        ).fetchone()
        if cursor is None:
            raise RuntimeError("Long-term memory cursor update did not return a row")
    return created_memories


def upsert_long_term_memory_embedding(
    memory_id: int,
    embedding_model: str,
    embedding: list[float],
) -> None:
    """幂等写入一条长期记忆的语义向量索引。

    Args:
        memory_id: 已存在的长期记忆主键。
        embedding_model: 生成向量的完整模型标识。
        embedding: 与数据库 VECTOR(768) 一致的有限浮点向量。
    Raises:
        ValueError: 模型标识为空、向量维度不正确或包含非有限值。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 在 SQL 前校验模型名、维度和浮点值，拒绝不兼容或不可检索的数据。
    # 2. 将 Python 浮点列表序列化为 pgvector 文本格式，避免依赖未安装的专用适配器。
    # 3. 使用 memory_id + embedding_model 的唯一键 upsert，使后台任务重试不会产生重复索引。
    # =========================================================================
    normalized_model = embedding_model.strip()
    if not normalized_model:
        raise ValueError("Embedding model cannot be empty")
    if len(embedding) != 768:
        raise ValueError("Embedding dimension must be 768")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in embedding):
        raise ValueError("Embedding values must be finite numbers")

    vector_literal = "[" + ",".join(format(float(value), ".12g") for value in embedding) + "]"
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO aiagent.aiagent_semantic_memories (
                memory_id, embedding_model, embedding
            )
            VALUES (%s, %s, %s::vector)
            ON CONFLICT (memory_id, embedding_model)
            DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (memory_id, normalized_model, vector_literal),
        )


def load_long_term_memory_recall_candidates(
    user_id: str,
    embedding_model: str,
    query_embedding: list[float],
    limit: int,
) -> list[LongTermMemoryRecallCandidate]:
    """按余弦距离粗查当前用户的有效长期记忆候选。

    Args:
        user_id: 当前用户标识，仅查询该用户的记忆。
        embedding_model: 必须与查询向量由同一模型生成。
        query_embedding: 与数据库 VECTOR(768) 一致的查询向量。
        limit: 粗查候选数量上限，必须为正数。
    Returns:
        由余弦距离升序排列的有效长期记忆候选。
    Raises:
        ValueError: 查询模型、向量或数量不符合数据库约束。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 复用写入侧的维度和数值约束，避免向 pgvector 发送无效查询向量。
    # 2. 仅连接当前用户、启用状态和相同 Embedding 模型的长期记忆。
    # 3. 用 pgvector 余弦距离进行粗排，精确业务分数由上层服务计算。
    # =========================================================================
    normalized_model = embedding_model.strip()
    if not normalized_model:
        raise ValueError("Embedding model cannot be empty")
    if limit < 1:
        raise ValueError("Long-term memory recall limit must be positive")
    if len(query_embedding) != 768:
        raise ValueError("Embedding dimension must be 768")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in query_embedding
    ):
        raise ValueError("Embedding values must be finite numbers")

    vector_literal = "[" + ",".join(format(float(value), ".12g") for value in query_embedding) + "]"
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT
                m.memory_id,
                m.content,
                m.memory_type,
                m.subject,
                m.attribute,
                m.importance,
                COALESCE(m.last_used_at, m.created_at) AS effective_last_used_at,
                s.embedding <=> %s::vector AS cosine_distance
            FROM aiagent.aiagent_semantic_memories AS s
            JOIN aiagent.aiagent_long_term_memories AS m ON m.memory_id = s.memory_id
            WHERE m.user_id = %s
              AND m.is_active = TRUE
              AND s.embedding_model = %s
            ORDER BY s.embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (vector_literal, user_id, normalized_model, vector_literal, limit),
        ).fetchall()
    return [
        LongTermMemoryRecallCandidate(
            memory_id=int(memory_id),
            content=str(content),
            memory_type=str(memory_type),
            subject=str(subject),
            attribute=str(attribute),
            importance=int(importance),
            effective_last_used_at=effective_last_used_at,
            cosine_distance=float(cosine_distance),
        )
        for (
            memory_id,
            content,
            memory_type,
            subject,
            attribute,
            importance,
            effective_last_used_at,
            cosine_distance,
        ) in rows
    ]


def load_active_long_term_memory_group(
    user_id: str,
    memory_type: str,
    subject: str,
    attribute: str,
) -> list[str]:
    """读取命中属性组中的全部有效长期记忆正文。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 使用向量命中行已有的结构化属性作为组标识，不从 content 猜测归属。
    # 2. 仅返回当前用户、当前属性组中仍有效的记录，避免已替代历史被重新注入。
    # 3. 按主键稳定排序，保证调试输出和后续 Prompt 内容顺序可预测。
    # =========================================================================
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT content
            FROM aiagent.aiagent_long_term_memories
            WHERE user_id = %s
              AND memory_type = %s
              AND subject = %s
              AND attribute = %s
              AND is_active = TRUE
            ORDER BY memory_id
            """,
            (user_id, memory_type, subject, attribute),
        ).fetchall()
    return [str(row[0]) for row in rows]


def load_thread_summary(thread_id: UUID, user_id: str) -> ThreadSummary | None:
    """读取用户所属会话的当前滚动摘要。"""

    with _connection() as connection:
        row = connection.execute(
            """
            SELECT s.summary, s.covered_to_seq, s.summary_version, s.summary_token_count
            FROM aiagent.aiagent_thread_summaries AS s
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = s.thread_id
            WHERE s.thread_id = %s AND t.user_id = %s
            """,
            (thread_id, user_id),
        ).fetchone()
    if row is None:
        return None
    return ThreadSummary(str(row[0]), int(row[1]), int(row[2]), int(row[3]))


def upsert_thread_summary(
    thread_id: UUID,
    user_id: str,
    summary: str,
    covered_to_seq: int,
    summary_token_count: int,
) -> bool:
    """写入更新后的摘要，仅允许覆盖范围向前推进。"""

    with _connection() as connection:
        row = connection.execute(
            """
            INSERT INTO aiagent.aiagent_thread_summaries (
                thread_id, summary, covered_to_seq, summary_version, summary_token_count
            )
            SELECT %s, %s, %s, 1, %s
            WHERE EXISTS (
                SELECT 1 FROM aiagent.aiagent_threads
                WHERE thread_id = %s AND user_id = %s
            )
            ON CONFLICT (thread_id) DO UPDATE
            SET summary = EXCLUDED.summary,
                covered_to_seq = EXCLUDED.covered_to_seq,
                summary_version = aiagent.aiagent_thread_summaries.summary_version + 1,
                summary_token_count = EXCLUDED.summary_token_count,
                updated_at = now()
            WHERE aiagent.aiagent_thread_summaries.covered_to_seq < EXCLUDED.covered_to_seq
            RETURNING thread_id
            """,
            (thread_id, summary, covered_to_seq, summary_token_count, thread_id, user_id),
        ).fetchone()
    return row is not None


def load_thread_task_state(thread_id: UUID, user_id: str) -> ThreadTaskState | None:
    """读取用户所属会话的审批任务状态。"""

    with _connection() as connection:
        row = connection.execute(
            """
            SELECT s.state, s.state_version
            FROM aiagent.aiagent_thread_states AS s
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = s.thread_id
            WHERE s.thread_id = %s AND t.user_id = %s
            """,
            (thread_id, user_id),
        ).fetchone()
    if row is None:
        return None
    state = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
    return ThreadTaskState(state, int(row[1]))


def set_pending_thread_actions(
    thread_id: UUID,
    user_id: str,
    pending_actions: list[dict[str, Any]],
) -> bool:
    """保存已校验的待确认 Action，并拒绝覆盖已有待确认状态。"""

    state = {"pending_actions": pending_actions, "approval_status": "pending", "approval_reason": None}
    with _connection() as connection:
        row = connection.execute(
            """
            WITH saved AS (
                INSERT INTO aiagent.aiagent_thread_states (thread_id, state, state_version)
                SELECT %s, %s::jsonb, 1
                WHERE EXISTS (
                    SELECT 1 FROM aiagent.aiagent_threads
                    WHERE thread_id = %s AND user_id = %s
                )
                ON CONFLICT (thread_id) DO UPDATE
                SET state = EXCLUDED.state,
                    state_version = aiagent.aiagent_thread_states.state_version + 1,
                    updated_at = now()
                WHERE aiagent.aiagent_thread_states.state->>'approval_status' <> 'pending'
                RETURNING thread_id
            )
            SELECT thread_id FROM saved
            UNION ALL
            SELECT s.thread_id
            FROM aiagent.aiagent_thread_states AS s
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = s.thread_id
            WHERE s.thread_id = %s
              AND t.user_id = %s
              AND s.state->>'approval_status' = 'pending'
            LIMIT 1
            """,
            (thread_id, json.dumps(state, ensure_ascii=False), thread_id, user_id, thread_id, user_id),
        ).fetchone()
    return row is not None


def resolve_thread_task_approval(
    thread_id: UUID,
    user_id: str,
    approved: bool,
    reason: str | None,
) -> bool:
    """将待审批状态原子推进为批准或拒绝。"""

    status = "approved" if approved else "rejected"
    with _connection() as connection:
        row = connection.execute(
            """
            UPDATE aiagent.aiagent_thread_states
            SET state = jsonb_set(
                    jsonb_set(state, '{approval_status}', to_jsonb(%s::text)),
                    '{approval_reason}', COALESCE(to_jsonb(%s::text), 'null'::jsonb)
                ),
                state_version = state_version + 1,
                updated_at = now()
            WHERE thread_id = %s
              AND (
                  state->>'approval_status' = 'pending'
                  OR (state->>'approval_status' = 'approved' AND %s = TRUE)
              )
              AND EXISTS (
                  SELECT 1 FROM aiagent.aiagent_threads
                  WHERE thread_id = %s AND user_id = %s
              )
            RETURNING thread_id
            """,
            (status, reason, thread_id, approved, thread_id, user_id),
        ).fetchone()
    return row is not None


def finish_thread_task(
    thread_id: UUID,
    user_id: str,
    status: str,
) -> bool:
    """结束批准、拒绝或执行失败后的待确认任务。"""

    if status not in {"completed", "failed", "rejected"}:
        raise ValueError(f"Unsupported task completion status: {status}")
    with _connection() as connection:
        row = connection.execute(
            """
            UPDATE aiagent.aiagent_thread_states
            SET state = jsonb_build_object('approval_status', %s::text),
                state_version = state_version + 1,
                updated_at = now()
            WHERE thread_id = %s
              AND state->>'approval_status' = ANY(%s)
              AND EXISTS (
                  SELECT 1 FROM aiagent.aiagent_threads
                  WHERE thread_id = %s AND user_id = %s
              )
            RETURNING thread_id
            """,
            (status, thread_id, ["approved", "rejected"], thread_id, user_id),
        ).fetchone()
    return row is not None


def list_threads(user_id: str) -> list[tuple[UUID, str | None, bool, Any]]:
    """返回用户会话标题及更新时间。"""
    with _connection() as connection:
        rows = connection.execute(
            """SELECT thread_id, title, title_is_custom, updated_at
               FROM aiagent.aiagent_threads WHERE user_id = %s
               ORDER BY updated_at DESC""",
            (user_id,),
        ).fetchall()
    return [(row[0], row[1], bool(row[2]), row[3]) for row in rows]


def update_thread_title(thread_id: UUID, user_id: str, title: str | None) -> bool:
    """更新自定义标题；传入空值时恢复自动标题。"""
    with _connection() as connection:
        row = connection.execute(
            """UPDATE aiagent.aiagent_threads SET title = %s, title_is_custom = %s, updated_at = now()
               WHERE thread_id = %s AND user_id = %s RETURNING thread_id""",
            (title, title is not None, thread_id, user_id),
        ).fetchone()
    return row is not None


def delete_thread(thread_id: UUID, user_id: str) -> bool:
    """删除指定用户拥有的会话，并由数据库级联清理关联消息。"""
    # [逻辑规划] 仅按会话 ID 和用户 ID 删除，避免用户删除其他用户的会话；
    # 数据库外键负责级联删除消息和摘要，并将长期记忆来源置空。
    with _connection() as connection:
        row = connection.execute(
            "DELETE FROM aiagent.aiagent_threads WHERE thread_id = %s AND user_id = %s RETURNING thread_id",
            (thread_id, user_id),
        ).fetchone()
    return row is not None


def set_auto_title_if_empty(thread_id: UUID, user_id: str, title: str) -> None:
    """首次提问时设置自动标题，不覆盖用户自定义标题。"""
    with _connection() as connection:
        connection.execute(
            """UPDATE aiagent.aiagent_threads SET title = %s, updated_at = now()
               WHERE thread_id = %s AND user_id = %s
                 AND title_is_custom = FALSE AND title IS NULL""",
            (title[:80], thread_id, user_id),
        )


def append_message(thread_id: UUID, user_id: str, role: str, content: str) -> int | None:
    """按会话序号追加一条消息，并返回分配的序号。"""

    with _connection() as connection:
        row = connection.execute(
            """
            UPDATE aiagent.aiagent_threads
            SET next_message_seq = next_message_seq + 1, updated_at = now()
            WHERE thread_id = %s AND user_id = %s
            RETURNING next_message_seq
            """,
            (thread_id, user_id),
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            """
            INSERT INTO aiagent.aiagent_messages (thread_id, seq, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (thread_id, row[0], role, content),
        )
    return int(row[0])
