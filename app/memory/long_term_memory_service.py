"""增量抽取并持久化用户长期记忆。"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from functools import partial
import json
import logging
import math
from typing import Any, Literal
from uuid import UUID

from anyio import to_thread
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, model_validator

from app.db.repositories import long_term_memory_repository
from app.db.models import LongTermMemoryInput, StoredMessage
from app.long_term_memory_policy import is_single_value_long_term_memory_attribute
from app.memory.embedding_service import embed_long_term_memory
from app.prompt_loader import (
    get_long_term_memory_conflict_prompt,
    get_long_term_memory_extraction_prompt,
)
from app.settings import Settings


logger = logging.getLogger(__name__)


# 同一线程可能并发触发多次抽取（例如一轮回复后、又补一轮）。
# 若不串行化，两次任务可能同时读游标、同时推进游标，导致重叠窗口重复抽取或游标错乱
@dataclass
class _ThreadExtractionLockState:
    """同一线程长期记忆抽取锁及其已登记任务数量。"""

    lock: asyncio.Lock
    users: int = 0


_thread_extraction_locks: dict[UUID, _ThreadExtractionLockState] = {}
_thread_extraction_locks_guard = asyncio.Lock()


@asynccontextmanager
async def _hold_thread_extraction_lock(thread_id: UUID) -> AsyncIterator[None]:
    """串行化同一线程的抽取任务，并在最后一个任务结束时释放锁状态。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 在全局保护锁内取得线程状态并增加引用计数，等待中的任务也必须计数。
    # 2. 获取线程锁后执行调用方逻辑，保证同一线程的游标读取和推进不重叠。
    # 3. 正常结束、异常和取消均进入 finally，减少计数；最后一个任务离开时删除状态。
    # =========================================================================
    async with _thread_extraction_locks_guard:
        state = _thread_extraction_locks.get(thread_id)
        if state is None:
            state = _ThreadExtractionLockState(lock=asyncio.Lock())
            _thread_extraction_locks[thread_id] = state
        state.users += 1
    try:
        async with state.lock:
            yield
    finally:
        async with _thread_extraction_locks_guard:
            state.users -= 1
            if state.users == 0 and _thread_extraction_locks.get(thread_id) is state:
                del _thread_extraction_locks[thread_id]

# 记忆类型
_ATTRIBUTES_BY_TYPE: dict[str, frozenset[str]] = {
    "profile": frozenset({"name", "residence", "occupation", "organization", "long_term_goal"}),
    "preference": frozenset(
        {"dietary_preference", "communication_preference", "work_preference"}
    ),
    "commitment": frozenset({"project_decision", "project_constraint"}),
}


class LongTermMemoryCandidate(BaseModel):
    """抽取模型返回的单条候选长期记忆。"""

    subject: Literal["user"]
    attribute: str = Field(min_length=1)
    content: str = Field(min_length=1)
    memory_type: Literal["profile", "preference", "commitment"]
    importance: int = Field(ge=0, le=10)
    evidence_message_seqs: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def attribute_must_match_memory_type(self) -> "LongTermMemoryCandidate":
        """确保抽取模型只能使用预定义的记忆属性。"""

        if self.attribute not in _ATTRIBUTES_BY_TYPE[self.memory_type]:
            raise ValueError("attribute does not match memory_type")
        return self


class LongTermMemoryExtraction(BaseModel):
    """抽取模型返回的候选记忆集合。"""

    memories: list[LongTermMemoryCandidate] = Field(default_factory=list)


class LongTermMemoryConflictDecision(BaseModel):
    """新记忆与相关旧记忆之间的结构化关系判断。"""

    action: Literal["ADD", "DUPLICATE", "UPDATE"]
    matched_memory_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def matched_ids_must_match_action(self) -> "LongTermMemoryConflictDecision":
        """确保新增操作不携带旧 ID，重复或更新操作必须明确目标。"""

        if self.action == "ADD" and self.matched_memory_ids:
            raise ValueError("ADD cannot contain matched memory IDs")
        if self.action != "ADD" and not self.matched_memory_ids:
            raise ValueError("DUPLICATE and UPDATE require matched memory IDs")
        return self


# 提取记忆使用的llm
def _build_memory_extraction_model(settings: Settings) -> ChatOpenAI:
    """创建不绑定工具的长期记忆抽取模型。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 复用现有 LLM 认证与基址配置，避免新增独立密钥来源。
    # 2. 使用独立模型、温度、超时和输出长度配置，隔离抽取成本与主 Agent。
    # 3. 不绑定工具，避免抽取调用拥有业务执行能力。
    # =========================================================================
    model_kwargs: dict[str, Any] = {
        "api_key": settings.llm_api_key.get_secret_value(),
        "model": settings.long_term_memory_extraction_model or settings.llm_model,
        "temperature": settings.long_term_memory_extraction_temperature,
        "timeout": settings.long_term_memory_extraction_timeout_seconds,
        "max_tokens": settings.long_term_memory_extraction_max_tokens,
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url
    extra_body = settings.llm_extra_body()
    if extra_body:
        model_kwargs["extra_body"] = extra_body
    return ChatOpenAI(**model_kwargs)


async def _resolve_memory_conflicts(
    user_id: str,
    memories: list[LongTermMemoryInput],
    embeddings: list[list[float] | None] | None,
    settings: Settings,
    resolver: Any | None = None,
) -> tuple[list[LongTermMemoryInput], list[list[float] | None] | None]:
    """识别多值记忆中的重复或状态更新，并绑定待停用的旧记忆 ID。

    Args:
        user_id: 当前用户标识。
        memories: 已通过证据与长度校验的新记忆。
        embeddings: 与 memories 同序的候选向量；Embedding 关闭时为 None。
        settings: 长期记忆冲突判断与模型配置。
        resolver: 测试时注入的结构化冲突判断模型。
    Returns:
        过滤重复并标记替代目标后的记忆及其同序向量。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 功能关闭、无向量或单值属性时保持现有流程；单值属性已有确定替代规则。
    # 2. 对每条多值记忆查询同组最相关旧记忆，避免把整个偏好组交给模型。
    # 3. 使用受控 schema 判断 ADD、DUPLICATE、UPDATE，并校验返回 ID 属于候选集合。
    # 4. DUPLICATE 丢弃新候选；UPDATE 绑定旧 ID；判断异常时保守保留为 ADD。
    # 5. 始终同步过滤向量列表，保证后续事务中的索引对应关系不发生错位。
    # =========================================================================
    if not getattr(settings, "long_term_memory_conflict_resolution_enabled", False):
        return memories, embeddings
    if embeddings is None:
        return memories, embeddings

    structured_resolver = resolver
    resolved_memories: list[LongTermMemoryInput] = []
    resolved_embeddings: list[list[float] | None] = []
    
    for memory, embedding in zip(memories, embeddings, strict=True):
        if embedding is None or is_single_value_long_term_memory_attribute(memory.attribute):
            resolved_memories.append(memory)
            resolved_embeddings.append(embedding)
            continue

        try:
            candidates = await to_thread.run_sync(
                long_term_memory_repository.load_long_term_memory_conflict_candidates,
                user_id,
                memory,
                settings.long_term_memory_embedding_model,
                embedding,
                settings.long_term_memory_conflict_candidate_limit,
            )
            if not candidates:
                resolved_memories.append(memory)
                resolved_embeddings.append(embedding)
                continue

            if structured_resolver is None:
                structured_resolver = _build_memory_extraction_model(settings).with_structured_output(
                    LongTermMemoryConflictDecision,
                    method="function_calling",
                )
            decision = await structured_resolver.ainvoke(
                [
                    SystemMessage(content=get_long_term_memory_conflict_prompt()),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "new_memory": memory.content,
                                "existing_memories": [
                                    {"memory_id": candidate.memory_id, "content": candidate.content}
                                    for candidate in candidates
                                ],
                            },
                            ensure_ascii=False,
                        )
                    ),
                ]
            )
            if not isinstance(decision, LongTermMemoryConflictDecision):
                raise ValueError("Conflict resolver returned an invalid result")
            candidate_ids = {candidate.memory_id for candidate in candidates}
            matched_ids = tuple(dict.fromkeys(decision.matched_memory_ids))
            if not set(matched_ids).issubset(candidate_ids):
                raise ValueError("Conflict resolver returned an unknown memory ID")
            if decision.action == "DUPLICATE":
                continue
            if decision.action == "UPDATE":
                memory = replace(memory, superseded_memory_ids=matched_ids)
        except Exception:
            logger.exception("long_term_memory_conflict_resolution_failed")

        resolved_memories.append(memory)
        resolved_embeddings.append(embedding)
    return resolved_memories, resolved_embeddings


def _messages_payload(messages: list[StoredMessage]) -> str:
    """将带序号的业务消息编码为抽取模型的受控输入。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 只发送业务消息的 seq、role 和 content，避免系统 Prompt 与执行状态泄露。
    # 2. 保留 seq，使模型输出可被服务端验证的消息级证据。
    # 3. 使用 JSON 编码，明确对话文本是数据而不是模型指令。
    # =========================================================================
    return json.dumps(
        {
            "messages": [
                {"seq": message.seq, "role": message.role, "content": message.content}
                for message in messages
            ]
        },
        ensure_ascii=False,
    )


def _validated_memory_inputs(
    extraction: LongTermMemoryExtraction,
    messages: list[StoredMessage],
    settings: Settings,
) -> list[LongTermMemoryInput]:
    """将模型候选校验为可写入数据库的长期记忆输入。"""

    # =========================================================================
    # [逻辑规划]
    # 1. 校验证据 seq 属于本轮输入，且每条长期记忆均有用户消息证据。
    # 2. 按配置过滤过低重要度和过长正文，避免低价值、不可控内容入库。
    # 3. 单条非法候选被跳过，不阻断其他有效候选。
    # 4. 有效候选超过单轮上限时，按重要度降序保留前 N 条，重要度相同者保持原序；
    #    截断而非整轮丢弃，避免用户并列表达多个对象时一条记忆都写不进。
    # =========================================================================
    messages_by_seq = {message.seq: message for message in messages}
    validated_inputs: list[LongTermMemoryInput] = []
    for candidate in extraction.memories:
        content = candidate.content.strip()
        evidence_message_seqs = tuple(sorted(set(candidate.evidence_message_seqs)))
        evidence_messages = [messages_by_seq.get(seq) for seq in evidence_message_seqs]
        if (
            not content
            or len(content) > settings.long_term_memory_max_content_length
            or candidate.importance < settings.long_term_memory_min_importance
            or len(evidence_message_seqs) > settings.long_term_memory_max_evidence_messages
            or any(message is None for message in evidence_messages)
            or not any(message is not None and message.role == "user" for message in evidence_messages)
        ):
            logger.warning("long_term_memory_candidate_rejected")
            continue
        validated_inputs.append(
            LongTermMemoryInput(
                content=content,
                memory_type=candidate.memory_type,
                subject=candidate.subject,
                attribute=candidate.attribute,
                importance=candidate.importance,
                evidence_message_seqs=evidence_message_seqs,
            )
        )

    limit = settings.long_term_memory_max_candidates
    if len(validated_inputs) > limit:
        logger.warning("long_term_memory_candidate_limit_truncated")
        validated_inputs = sorted(
            validated_inputs, key=lambda memory: memory.importance, reverse=True
        )[:limit]
    return validated_inputs


def _validated_candidate_embedding(embedding: object, settings: Settings) -> list[float]:
    """校验候选 Embedding，避免异常向量中断原文记忆入库。

    Args:
        embedding: Embedding 服务返回的候选向量。
        settings: 包含预期向量维度的运行配置。
    Returns:
        可安全传入数据库事务的有限浮点向量。
    Raises:
        ValueError: 返回类型、维度或数值不符合向量索引约束。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 只接受列表，避免第三方 Embedding 实现返回不可预测的对象。
    # 2. 校验维度与数据库 VECTOR(768) 配置一致。
    # 3. 将数值规范为 float 并拒绝 NaN、Infinity；调用方将异常降级为 None。
    # =========================================================================
    if not isinstance(embedding, list):
        raise ValueError("Embedding must be a list")
    if len(embedding) != settings.long_term_memory_embedding_dimensions:
        raise ValueError("Embedding dimension does not match configured dimensions")
    normalized_embedding = [float(value) for value in embedding]
    if not all(math.isfinite(value) for value in normalized_embedding):
        raise ValueError("Embedding values must be finite")
    return normalized_embedding


async def extract_and_persist_long_term_memories(
    thread_id: UUID,
    user_id: str,
    *,
    settings: Settings,
    extractor: Any | None = None,
    embedder: Any | None = None,
    conflict_resolver: Any | None = None,
    through_seq: int | None = None,
) -> int:
    """从最近完成的对话增量抽取候选记忆并受控写入数据库。

    Args:
        thread_id: 刚完成回复的会话标识。
        user_id: 会话所属用户，用于隔离消息和长期记忆。
        settings: 已校验的运行配置。
        extractor: 测试时注入的已结构化输出模型；默认创建生产模型。
        embedder: 测试时注入的向量生成函数；默认使用本地 BGE 模型。
        conflict_resolver: 测试时注入的新旧记忆关系判断模型。
    Returns:
        实际新增的长期记忆数量；抽取或校验失败时返回 0。
    """

    # =========================================================================
    # [逻辑规划]
    # 1. 功能关闭时直接返回，避免任何额外数据库或模型调用。
    # 2. 读取上次游标之后的新增业务消息，避免重叠窗口重复抽取。
    # 3. 调用无工具的结构化抽取模型；模型或解析失败时记录异常并降级为空结果。
    # 4. 对候选执行服务层校验后，在数据库事务内去重、替代和写入证据。
    # 5. 仅为本次新增记忆生成向量；Embedding 失败不回滚已持久化的原文和证据。
    # 6. 此函数的异常不会向聊天主流程传播，保证主回复已完成后仍可正常交付。
    # =========================================================================

    if not settings.long_term_memory_enabled or through_seq is None:
        return 0

    async with _hold_thread_extraction_lock(thread_id):
        try:
            batch = await to_thread.run_sync(
                long_term_memory_repository.load_incremental_long_term_memory_messages,
                thread_id,
                user_id,
                through_seq,
            )

            if batch is None:
                return 0

            messages, _processed_seq = batch
            if not messages:
                return 0

            structured_extractor = extractor
            if structured_extractor is None:
                structured_extractor = _build_memory_extraction_model(settings).with_structured_output(
                    LongTermMemoryExtraction,
                    method="function_calling",
                )
            result = await structured_extractor.ainvoke(
                [
                    SystemMessage(content=get_long_term_memory_extraction_prompt()),
                    HumanMessage(content=_messages_payload(messages)),
                ]
            )
            if not isinstance(result, LongTermMemoryExtraction):
                logger.warning("long_term_memory_extraction_invalid_result")
                return 0

            memory_inputs: list[LongTermMemoryInput] = _validated_memory_inputs(result, messages, settings)

            persisted_memories = []

            if memory_inputs:
                candidate_embeddings: list[list[float] | None] | None = None
                if settings.long_term_memory_embedding_enabled:
                    embedding_function = embedder or embed_long_term_memory
                    candidate_embeddings = []

                    for memory in memory_inputs:
                        try:
                            embedding = await asyncio.wait_for(
                                to_thread.run_sync(
                                    embedding_function,
                                    memory.content,
                                    settings,
                                    abandon_on_cancel=True,
                                ),
                                timeout=settings.long_term_memory_embedding_timeout_seconds,
                            )
                            embedding = _validated_candidate_embedding(embedding, settings)
                        except Exception:
                            logger.exception("long_term_memory_embedding_failed")
                            embedding = None

                        candidate_embeddings.append(embedding)

                memory_inputs, candidate_embeddings = await _resolve_memory_conflicts(
                    user_id,
                    memory_inputs,
                    candidate_embeddings,
                    settings,
                    conflict_resolver,
                )

                if memory_inputs:
                    persisted_memories = await to_thread.run_sync(
                        partial(
                            long_term_memory_repository.merge_long_term_memories,
                            thread_id,
                            user_id,
                            memory_inputs,
                            through_seq,
                            memory_embeddings=candidate_embeddings,
                            embedding_model=settings.long_term_memory_embedding_model,
                            semantic_dedup_enabled=(
                                settings.long_term_memory_semantic_dedup_enabled
                                and settings.long_term_memory_embedding_enabled
                            ),
                            semantic_dedup_distance=settings.long_term_memory_semantic_dedup_distance,
                        )
                    )
                else:
                    cursor_advanced = await to_thread.run_sync(
                        long_term_memory_repository.advance_long_term_memory_cursor,
                        thread_id,
                        user_id,
                        through_seq,
                    )
                    if not cursor_advanced:
                        raise RuntimeError("Long-term memory cursor update failed")
            else:
                cursor_advanced = await to_thread.run_sync(
                    long_term_memory_repository.advance_long_term_memory_cursor,
                    thread_id,
                    user_id,
                    through_seq,
                )
                if not cursor_advanced:
                    raise RuntimeError("Long-term memory cursor update failed")
            return len(persisted_memories)
        except Exception:
            logger.exception("long_term_memory_processing_failed")
            return 0
