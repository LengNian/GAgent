"""PostgreSQL 持久化操作。"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
from typing import Any
from uuid import UUID

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


def append_message(thread_id: UUID, user_id: str, role: str, content: str) -> bool:
    """按会话序号追加一条消息，并更新会话时间。"""

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
            return False
        connection.execute(
            """
            INSERT INTO aiagent.aiagent_messages (thread_id, seq, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (thread_id, row[0], role, content),
        )
    return True
