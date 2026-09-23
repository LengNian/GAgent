"""聊天模型运行时构造。"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.settings import Settings


def build_model(settings: Settings) -> BaseChatModel:
    """根据已校验配置创建 OpenAI 兼容聊天模型。"""

    model_kwargs: dict[str, Any] = {
        "api_key": settings.llm_api_key.get_secret_value(),
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_seconds,
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url
    extra_body = settings.llm_extra_body()
    if extra_body:
        model_kwargs["extra_body"] = extra_body
    return ChatOpenAI(**model_kwargs)
