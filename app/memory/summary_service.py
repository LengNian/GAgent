"""按需维护会话滚动摘要并组装模型上下文。"""

import json
import logging
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from anyio import to_thread
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app import database
from app.context import ContextCompiler, TokenCounter
from app.database import StoredMessage, ThreadSummary
from app.prompt_loader import get_thread_summary_prompt
from app.settings import Settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompiledThreadContext:
    """最终模型上下文及本轮摘要是否成功推进。"""

    messages: list[BaseMessage]
    # 本轮是否真的/推进了摘要，用于读写数据库的判断
    summary_updated: bool


class _PrefixedTokenCounter(TokenCounter):
    """将固定摘要一并纳入原文窗口的聊天模板计数。"""

    def __init__(self, token_counter: TokenCounter, prefix: SystemMessage) -> None:
        self._token_counter = token_counter
        self._prefix = prefix

    def count_text(self, text: str) -> int:
        """复用当前模型的正文 token 计算方式。"""

        return self._token_counter.count_text(text)

    def count_message(self, message: BaseMessage) -> int:
        """复用当前模型的单条正文 token 计算方式。"""

        return self._token_counter.count_message(message)

    def count_messages(self, messages: Sequence[BaseMessage]) -> int:
        """计算固定摘要与候选原文共同使用的聊天模板 token。"""

        return self._token_counter.count_messages([self._prefix, *messages])


def _to_message(stored_message: StoredMessage) -> BaseMessage:
    """将持久化业务消息转换为 LangChain 消息。"""

    if stored_message.role == "user":
        return HumanMessage(content=stored_message.content)
    if stored_message.role == "assistant":
        return AIMessage(content=stored_message.content)
    raise ValueError(f"Unsupported stored message role: {stored_message.role}")


# 创建进行摘要总结的模型
def _build_summary_model(settings: Settings) -> ChatOpenAI:
    """创建不绑定工具的摘要模型。"""

    model_kwargs: dict[str, Any] = {
        "api_key": settings.llm_api_key.get_secret_value(),
        "model": settings.llm_model,
        "temperature": 0,
        "timeout": settings.llm_timeout_seconds,
        "max_tokens": settings.context_summary_max_tokens,
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**model_kwargs)


def _compile_window(
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    settings: Settings,
    token_counter: TokenCounter,
):
    """使用统一规则选择最近原文窗口。"""

    return ContextCompiler(
        max_tokens=max_tokens,
        max_message_tokens=settings.context_max_message_tokens,
        max_messages=settings.context_max_messages,
        token_counter=token_counter,
    ).compile(messages)


async def _generate_summary(
    existing_summary: ThreadSummary | None,
    messages: list[StoredMessage],
    *,
    settings: Settings,
    model: ChatOpenAI | None = None,
) -> str | None:
    """根据旧摘要和新增旧消息生成新的完整摘要。"""

    payload = {
        "existing_summary": existing_summary.summary if existing_summary else "",
        "messages": [{"role": message.role, "content": message.content} for message in messages],
    }
    try:
        response = await (model or _build_summary_model(settings)).ainvoke(
            [
                SystemMessage(content=get_thread_summary_prompt()),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
            ]
        )
    except Exception:
        logger.exception("Thread summary generation failed")
        return None
    content = getattr(response, "content", "")
    if not isinstance(content, str) or not content.strip():
        return None

    print("\n*********************Summary********************************")
    print(content.strip())
    print("************************888*********************************\n")

    return content.strip()



# 将已生成的摘要包装成一条系统信息
def _summary_context_message(summary: ThreadSummary) -> SystemMessage:
    """将已验证摘要标注为上下文，而不是 Agent 系统 Prompt。"""

    return SystemMessage(
        content=f"会话历史摘要：\n{summary.summary}",
        additional_kwargs={"context_kind": "thread_summary"},
    )


def _summary_budget_message(settings: Settings, token_counter: TokenCounter) -> SystemMessage:
    """构造摘要最大输出预算的占位消息，用于预留完整模板空间。"""

    content = "中" * settings.context_summary_max_tokens
    while token_counter.count_text(content) < settings.context_summary_max_tokens:
        content += content
    return SystemMessage(content=f"会话历史摘要：\n{content}")


async def compile_thread_context(
    thread_id: UUID,
    user_id: str,
    stored_messages: list[StoredMessage],
    summary: ThreadSummary | None,
    *,
    settings: Settings,
    token_counter: TokenCounter,
) -> CompiledThreadContext:
    """必要时更新摘要，并返回符合总预算的最终业务上下文。"""

    covered_to_seq = summary.covered_to_seq if summary else 0
    unsummarized = [message for message in stored_messages if message.seq > covered_to_seq]
    raw_messages = [_to_message(message) for message in unsummarized]
    context_prefix = _summary_context_message(summary) if summary is not None else None
    context_messages = [*([context_prefix] if context_prefix is not None else []), *raw_messages]
    context_tokens = token_counter.count_messages(context_messages)

    # 临时查看摘要触发余量时，只需注释或取消注释这一处输出。
    print(
        "=== context budget ===\n"
        f"tokens: {context_tokens}/{settings.context_max_tokens}, "
        f"remaining: {settings.context_max_tokens - context_tokens}\n"
        f"unsummarized messages: {len(raw_messages)}/{settings.context_max_messages}, "
        f"remaining: {settings.context_max_messages - len(raw_messages)}\n"
        "======================",
        flush=True,
    )

    preliminary_counter = (
        _PrefixedTokenCounter(token_counter, context_prefix) if context_prefix is not None else token_counter
    )

    preliminary = _compile_window(
        raw_messages,
        max_tokens=settings.context_max_tokens,
        settings=settings,
        token_counter=preliminary_counter,
    )

    needs_summary = preliminary.dropped_message_count > 0
    active_summary = summary
    summary_updated = False

    if needs_summary:
        summary_budget_counter = _PrefixedTokenCounter(
            token_counter,
            _summary_budget_message(settings, token_counter),
        )

        reduced_window = _compile_window(
            raw_messages,
            max_tokens=settings.context_max_tokens,
            settings=settings,
            token_counter=summary_budget_counter,
        )
        
        # 未摘要数 - reduced_window 条数
        summarized_count = len(unsummarized) - len(reduced_window.messages)

        messages_to_summarize = unsummarized[:summarized_count]
        
        if messages_to_summarize:
            generated_summary = await _generate_summary(
                summary,
                messages_to_summarize,
                settings=settings,
            )
            if generated_summary is not None:
                covered_to_seq = messages_to_summarize[-1].seq
                summary_token_count = token_counter.count_text(generated_summary)
                persisted = await to_thread.run_sync(
                    database.upsert_thread_summary,
                    thread_id,
                    user_id,
                    generated_summary,
                    covered_to_seq,
                    summary_token_count,
                )
                if persisted:
                    active_summary = ThreadSummary(
                        generated_summary,
                        covered_to_seq,
                        (summary.summary_version + 1) if summary else 1,
                        summary_token_count,
                    )
                    summary_updated = True
        raw_window = reduced_window
    else:
        raw_window = preliminary

    if active_summary is not None:
        if summary_updated:
            raw_messages = [
                _to_message(message)
                for message in stored_messages
                if message.seq > active_summary.covered_to_seq
            ]
        context_counter = _PrefixedTokenCounter(
            token_counter,
            _summary_context_message(active_summary),
        )
        raw_window = _compile_window(
            raw_messages,
            max_tokens=settings.context_max_tokens,
            settings=settings,
            token_counter=context_counter,
        )
        return CompiledThreadContext(
            [_summary_context_message(active_summary), *raw_window.messages],
            summary_updated,
        )
    return CompiledThreadContext(raw_window.messages, summary_updated)
