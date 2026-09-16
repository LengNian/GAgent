"""业务 PostgreSQL 连接池和事务上下文。"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from app.settings import get_settings

_pool: Any | None = None


def get_pool() -> Any:
    """获取进程级连接池，首次使用时延迟创建。"""

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
def connection() -> Iterator[Any]:
    """借用连接并在退出时提交或回滚事务。"""

    with get_pool().connection() as database_connection:
        try:
            yield database_connection
            database_connection.commit()
        except Exception:
            database_connection.rollback()
            raise


def close_pool() -> None:
    """关闭进程级连接池。"""

    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None

