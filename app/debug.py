"""本地调试输出：实际发送给模型的对话消息。"""

from typing import Any

from langchain_core.messages import SystemMessage


def print_model_messages(messages: list[Any]) -> None:
    """输出本次请求发送给模型的业务上下文，不输出固定 Prompt。"""

    print("=== model messages ===", flush=True)
    for index, message in enumerate(messages, start=1):
        # 仅允许输出摘要和长期记忆两类业务 SystemMessage，过滤 Agent Prompt。
        is_context = message.additional_kwargs.get("context_kind") in {
            "thread_summary",
            "long_term_memory",
        }
        if isinstance(message, SystemMessage) and not is_context:
            continue
        print(
            index,
            type(message).__name__,
            repr(getattr(message, "content", None)),
            flush=True,
        )
