"""配置化加载模型官方 tokenizer。"""

from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.context.compiler import TokenCounter
from app.settings import Settings


class HuggingFaceTokenCounter(TokenCounter):
    """使用 Hugging Face 官方模型仓库提供的 tokenizer 计算 token。"""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    # 重写count_text方法，不估算，直接用官方tokenizer计算
    def count_text(self, text: str) -> int:
        """按官方 tokenizer 计算文本 token 数，不添加聊天特殊标记。"""

        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def count_messages(self, messages: list[BaseMessage]) -> int:
        """按模型聊天模板计算消息 token 数，包含角色和生成提示开销。"""

        chat_messages = [self._to_chat_message(message) for message in messages]
        token_ids = self.tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(token_ids)

    @staticmethod
    def _to_chat_message(message: BaseMessage) -> dict[str, str]:
        """将当前纯文本 LangChain 消息转换为聊天模板支持的角色和内容。"""

        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            raise ValueError(f"Unsupported message type for tokenizer: {type(message).__name__}")
        if not isinstance(message.content, str):
            raise ValueError("Only text messages are supported by the current context tokenizer")
        return {"role": role, "content": message.content}


def create_token_counter(settings: Settings) -> TokenCounter:
    """根据配置返回当前模型对应的官方 tokenizer 计数器。"""

    if settings.llm_tokenizer_backend != "huggingface":
        raise ValueError(f"Unsupported tokenizer backend: {settings.llm_tokenizer_backend}")
    return _create_huggingface_token_counter(
        settings.llm_tokenizer_name,
        settings.llm_tokenizer_revision,
        settings.llm_tokenizer_trust_remote_code,
    )


@lru_cache(maxsize=4)
def _create_huggingface_token_counter(
    tokenizer_name: str,
    tokenizer_revision: str | None,
    trust_remote_code: bool,
) -> HuggingFaceTokenCounter:
    """加载并缓存指定官方 tokenizer；初始化失败时阻止使用不准确的估算器。"""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for the configured tokenizer") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=tokenizer_revision,
            trust_remote_code=trust_remote_code,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to load tokenizer: {tokenizer_name}") from error
    return HuggingFaceTokenCounter(tokenizer)
