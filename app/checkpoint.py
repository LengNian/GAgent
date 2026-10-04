"""LangGraph PostgreSQL checkpoint 生命周期管理。"""

from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_stack: AsyncExitStack | None = None
_checkpointer: Any | None = None


async def open_checkpointer(database_url: str | None) -> Any | None:
    """初始化 PostgreSQL checkpoint；未配置数据库时保持禁用。"""
    global _stack, _checkpointer
    if not database_url:
        return None
    _stack = AsyncExitStack()
    await _stack.__aenter__()
    # 使用连接池而非单连接：单条 AsyncConnection 空闲被 DB/网络回收后无法自愈，
    # 会让后续所有请求持续报 "the connection is closed"；池子取连接时会探活并重连。
    # kwargs 复刻 from_conn_string 隐含的连接参数，其中 autocommit=True 是 checkpoint
    # 写入提交的前提，缺失会让写入卡在未提交事务。
    pool = await _stack.enter_async_context(
        AsyncConnectionPool(
            conninfo=database_url,
            min_size=1,
            max_size=10,
            open=False,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
    )
    _checkpointer = AsyncPostgresSaver(conn=pool)
    await _checkpointer.setup()
    return _checkpointer


def get_checkpointer() -> Any | None:
    """返回当前进程共享的 checkpoint 实例。"""
    return _checkpointer


async def close_checkpointer() -> None:
    """释放 checkpoint 数据库连接。"""
    global _stack, _checkpointer
    if _stack is not None:
        await _stack.aclose()
    _stack = None
    _checkpointer = None
