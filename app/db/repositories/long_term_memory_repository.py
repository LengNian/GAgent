"""长期记忆、证据和语义向量仓储。"""

import math
from uuid import UUID

from app.db.connection import connection as _connection
from app.db.models import (
    LongTermMemoryInput,
    LongTermMemoryRecallCandidate,
    PersistedLongTermMemory,
    StoredMessage,
)
from app.long_term_memory_policy import is_single_value_long_term_memory_attribute
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


def load_long_term_memory_conflict_candidates(
    user_id: str,
    memory: LongTermMemoryInput,
    embedding_model: str,
    embedding: list[float],
    limit: int,
) -> list[PersistedLongTermMemory]:
    """读取与新候选同组且语义最接近的有效旧记忆。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 校验模型、向量和数量，避免向 pgvector 发送不兼容参数。
    # 2. 只查询当前用户、同类型、同主体、同属性且仍有效的记忆。
    # 3. 按余弦距离取少量候选，交由上层模型判断重复或状态冲突。
    # =========================================================================
    normalized_model = embedding_model.strip()
    if not normalized_model:
        raise ValueError("Embedding model cannot be empty")
    if limit < 1:
        raise ValueError("Conflict candidate limit must be positive")
    if len(embedding) != 768 or not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in embedding
    ):
        raise ValueError("Memory embedding must contain 768 finite values")


    vector_literal = "[" + ",".join(format(float(value), ".12g") for value in embedding) + "]"
    
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT m.memory_id, m.content
            FROM aiagent.aiagent_semantic_memories AS s
            JOIN aiagent.aiagent_long_term_memories AS m ON m.memory_id = s.memory_id
            WHERE m.user_id = %s
              AND m.memory_type = %s
              AND m.subject = %s
              AND m.attribute = %s
              AND m.is_active = TRUE
              AND s.embedding_model = %s
            ORDER BY s.embedding <=> %s::vector ASC
            LIMIT %s
            """,
            (
                user_id,
                memory.memory_type,
                memory.subject,
                memory.attribute,
                normalized_model,
                vector_literal,
                limit,
            ),
        ).fetchall()
    return [PersistedLongTermMemory(int(memory_id), str(content)) for memory_id, content in rows]


##
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
    # 3. 校验冲突模型返回的旧记忆 ID 只属于当前用户和同一属性组。
    # 4. 活跃记忆正文完全相同时跳过；无明确替代关系时再执行语义去重。
    # 5. 单值属性继续替代同属性旧值，多值属性只停用冲突模型明确匹配的旧值。
    # 6. 同一事务写入原文、向量、证据、旧值状态和游标，任一步失败均整体回滚。
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

            explicit_replaced_memory_ids: list[int] = []
            if memory.superseded_memory_ids:
                replacement_rows = connection.execute(
                    """
                    SELECT memory_id
                    FROM aiagent.aiagent_long_term_memories
                    WHERE memory_id = ANY(%s)
                      AND user_id = %s
                      AND memory_type = %s
                      AND subject = %s
                      AND attribute = %s
                      AND is_active = TRUE
                    FOR UPDATE
                    """,
                    (
                        list(memory.superseded_memory_ids),
                        user_id,
                        memory.memory_type,
                        memory.subject,
                        memory.attribute,
                    ),
                ).fetchall()
                explicit_replaced_memory_ids = [int(row[0]) for row in replacement_rows]
                if set(explicit_replaced_memory_ids) != set(memory.superseded_memory_ids):
                    raise ValueError("Superseded long-term memory IDs are invalid")

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

            # 证据信息校验
            evidence_by_seq = {int(seq): str(role) for seq, role in evidence_rows}
            expected_evidence = set(memory.evidence_message_seqs)
            if set(evidence_by_seq) != expected_evidence or "user" not in evidence_by_seq.values():
                raise ValueError("Long-term memory evidence is invalid")

            # 去重 正文完全相同且仍生效的记忆存在就去掉
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
                if explicit_replaced_memory_ids:
                    connection.execute(
                        """
                        UPDATE aiagent.aiagent_long_term_memories
                        SET is_active = FALSE,
                            inactive_reason = 'superseded',
                            inactivated_at = now()
                        WHERE memory_id = ANY(%s)
                        """,
                        (explicit_replaced_memory_ids,),
                    )
                continue

            # 语义去重（仅多值属性）
            if (
                semantic_dedup_enabled
                and embedding is not None
                and not explicit_replaced_memory_ids
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

                # 要在同组中进行筛选，即 memory_type、subject、attribute 需要相同，计算余弦相似度要大于semantic_dedup_distance
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

            # 单值属性，替换旧的记忆
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
            replaced_memory_ids = list(
                dict.fromkeys(explicit_replaced_memory_ids + [int(row[0]) for row in replaced_rows])
            )


            # 插入新记忆
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

# 计算输入query与embedding的粗召回
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
        # <=> 是pgvector的余弦距离算子
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


# 筛选is_active为True的长期记忆
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
