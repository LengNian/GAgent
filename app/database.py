"""PostgreSQL 持久化操作。"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from app.settings import get_settings


@contextmanager
def _connection() -> Iterator[Any]:
    """创建一次性数据库连接，并在退出时提交或回滚事务。"""

    database_url = get_settings().database_url
    if not database_url:
        raise RuntimeError("DATABASE_URL is not configured")
    try:
        import psycopg
    except ImportError as error:
        raise RuntimeError("PostgreSQL driver is missing; install psycopg[binary]") from error

    connection = psycopg.connect(database_url)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def create_thread(thread_id: UUID, user_id: str) -> None:
    """持久化新会话。"""

    with _connection() as connection:
        connection.execute(
            "INSERT INTO aiagent.threads (thread_id, user_id) VALUES (%s, %s)",
            (thread_id, user_id),
        )


def thread_exists_for_user(thread_id: UUID, user_id: str) -> bool:
    """判断会话是否存在且属于指定用户。"""

    with _connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM aiagent.threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        ).fetchone()
    return row is not None


def load_messages(thread_id: UUID, user_id: str) -> list[tuple[str, str]] | None:
    """读取用户所属会话的消息；会话不存在或不属于用户时返回 None。"""

    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT m.role, m.content
            FROM aiagent.messages AS m
            JOIN aiagent.threads AS t ON t.thread_id = m.thread_id
            WHERE m.thread_id = %s AND t.user_id = %s
            ORDER BY m.seq
            """,
            (thread_id, user_id),
        ).fetchall()
        thread = connection.execute(
            "SELECT 1 FROM aiagent.threads WHERE thread_id = %s AND user_id = %s",
            (thread_id, user_id),
        ).fetchone()
    if thread is None:
        return None
    return [(str(role), str(content)) for role, content in rows]


def append_message(thread_id: UUID, user_id: str, role: str, content: str) -> bool:
    """按会话序号追加一条消息，并更新会话时间。"""

    with _connection() as connection:
        row = connection.execute(
            """
            UPDATE aiagent.threads
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
            INSERT INTO aiagent.messages (thread_id, seq, role, content)
            VALUES (%s, %s, %s, %s)
            """,
            (thread_id, row[0], role, content),
        )
    return True
