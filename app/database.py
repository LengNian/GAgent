"""PostgreSQL 持久化操作。"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import UUID

from app.settings import get_settings

_pool: Any | None = None


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


def list_threads(user_id: str) -> list[tuple[UUID, str | None, bool, Any]]:
    """返回用户会话标题及更新时间。"""
    with _connection() as connection:
        rows = connection.execute(
            """SELECT thread_id, title, title_is_custom, updated_at
               FROM aiagent.threads WHERE user_id = %s
               ORDER BY updated_at DESC""",
            (user_id,),
        ).fetchall()
    return [(row[0], row[1], bool(row[2]), row[3]) for row in rows]


def update_thread_title(thread_id: UUID, user_id: str, title: str | None) -> bool:
    """更新自定义标题；传入空值时恢复自动标题。"""
    with _connection() as connection:
        row = connection.execute(
            """UPDATE aiagent.threads SET title = %s, title_is_custom = %s, updated_at = now()
               WHERE thread_id = %s AND user_id = %s RETURNING thread_id""",
            (title, title is not None, thread_id, user_id),
        ).fetchone()
    return row is not None


def set_auto_title_if_empty(thread_id: UUID, user_id: str, title: str) -> None:
    """首次提问时设置自动标题，不覆盖用户自定义标题。"""
    with _connection() as connection:
        connection.execute(
            """UPDATE aiagent.threads SET title = %s, updated_at = now()
               WHERE thread_id = %s AND user_id = %s
                 AND title_is_custom = FALSE AND title IS NULL""",
            (title[:80], thread_id, user_id),
        )


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
