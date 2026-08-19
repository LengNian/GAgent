"""从环境文件加载并校验应用配置。"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """创建聊天模型所需的、经过 Pydantic 校验的配置。"""

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_api_key: SecretStr = Field(validation_alias="LLM_API_KEY")
    llm_model: str = Field(validation_alias="LLM_MODEL")
    llm_base_url: str | None = Field(default=None, validation_alias="LLM_BASE_URL")
    llm_temperature: float = Field(validation_alias="LLM_TEMPERATURE")
    llm_timeout_seconds: float = Field(validation_alias="LLM_TIMEOUT_SECONDS")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存当前进程使用的配置实例。

    逻辑规划：
    1. 首次调用时由 Pydantic Settings 读取 config/.env 和环境变量。
    2. 对必填字段和字段类型执行统一校验；校验失败直接抛出异常。
    3. 使用 lru_cache 复用同一实例，避免每次创建 Agent 都重新读取配置。
    """

    return Settings()
