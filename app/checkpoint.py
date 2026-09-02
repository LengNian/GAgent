"""LangGraph PostgreSQL checkpoint 生命周期管理。"""

from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

_stack: AsyncExitStack | None = None
_checkpointer: Any | None = None


async def open_checkpointer(database_url: str | None) -> Any | None:
    """初始化 PostgreSQL checkpoint；未配置数据库时保持禁用。"""
    global _stack, _checkpointer
    if not database_url:
        return None
    _stack = AsyncExitStack()
    await _stack.__aenter__()
    _checkpointer = await _stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(database_url)
    )
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
