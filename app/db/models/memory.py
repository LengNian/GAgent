"""会话、摘要和长期记忆的数据对象。"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class StoredMessage:
    """带持久化序号的业务消息。"""

    seq: int
    role: str
    content: str
    emotion: str | None = None


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
    superseded_memory_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PersistedLongTermMemory:
    """本次事务中新建的长期记忆。"""

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
