"""会话和消息仓储。"""

from typing import Any
from uuid import UUID

from app.db.connection import connection as _connection
from app.db.models import StoredMessage


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


def load_stored_messages(thread_id: UUID, user_id: str) -> list[StoredMessage] | None:
    """读取带序号的业务消息。"""
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


def load_messages(thread_id: UUID, user_id: str) -> list[tuple[str, str]] | None:
    """读取用户所属会话的消息。"""
    stored_messages = load_stored_messages(thread_id, user_id)
    if stored_messages is None:
        return None
    return [(message.role, message.content) for message in stored_messages]


def load_message(thread_id: UUID, user_id: str, sequence: int) -> StoredMessage | None:
    """读取用户拥有的单条业务消息。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 以 thread_id、user_id 与序号联合查询，避免跨用户读取待播报文本。
    # 2. 未命中时返回 None，由 API 统一转换为 404。
    # =========================================================================
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT m.seq, m.role, m.content
            FROM aiagent.aiagent_messages AS m
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = m.thread_id
            WHERE m.thread_id = %s AND t.user_id = %s AND m.seq = %s
            """,
            (thread_id, user_id, sequence),
        ).fetchone()
    if row is None:
        return None
    return StoredMessage(int(row[0]), str(row[1]), str(row[2]))


def load_recent_stored_messages(thread_id: UUID, user_id: str, limit: int) -> list[StoredMessage] | None:
    """读取当前会话最近的业务消息，并恢复 seq 正序。"""
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


def append_message(thread_id: UUID, user_id: str, role: str, content: str) -> int | None:
    """按会话序号追加消息。"""
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
    """更新自定义标题。"""
    with _connection() as connection:
        row = connection.execute(
            """UPDATE aiagent.aiagent_threads SET title = %s, title_is_custom = %s, updated_at = now()
               WHERE thread_id = %s AND user_id = %s RETURNING thread_id""",
            (title, title is not None, thread_id, user_id),
        ).fetchone()
    return row is not None


def delete_thread(thread_id: UUID, user_id: str) -> bool:
    """删除指定用户拥有的会话。"""
    with _connection() as connection:
        row = connection.execute(
            "DELETE FROM aiagent.aiagent_threads WHERE thread_id = %s AND user_id = %s RETURNING thread_id",
            (thread_id, user_id),
        ).fetchone()
    return row is not None


def set_auto_title_if_empty(thread_id: UUID, user_id: str, title: str) -> None:
    """首次提问时设置自动标题。"""
    with _connection() as connection:
        connection.execute(
            """UPDATE aiagent.aiagent_threads SET title = %s, updated_at = now()
               WHERE thread_id = %s AND user_id = %s
                 AND title_is_custom = FALSE AND title IS NULL""",
            (title[:80], thread_id, user_id),
        )

__all__ = [
    "create_thread",
    "thread_exists_for_user",
    "load_messages",
    "load_message",
    "load_stored_messages",
    "load_recent_stored_messages",
    "append_message",
    "list_threads",
    "update_thread_title",
    "delete_thread",
    "set_auto_title_if_empty",
]
