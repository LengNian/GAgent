"""本地调试输出：实际发送给模型的对话消息。"""

from typing import Any

from langchain_core.messages import SystemMessage


def print_model_messages(messages: list[Any]) -> None:
    """输出本次请求发送给模型的非 Prompt 对话消息。"""

    print("=== model messages ===", flush=True)
    for index, message in enumerate(messages, start=1):
        is_context = message.additional_kwargs.get("context_kind") == "thread_summary"
        if isinstance(message, SystemMessage) and not is_context:
            continue
        print(
            index,
            type(message).__name__,
            repr(getattr(message, "content", None)),
            flush=True,
        )
