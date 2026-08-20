"""从环境文件加载并校验应用配置。"""

from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
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
    nms_api_base_url: str = Field(validation_alias="NMS_API_BASE_URL")

    @field_validator("nms_api_base_url")
    @classmethod
    def nms_api_base_url_must_use_http(cls, value: str) -> str:
        """校验 NMS 服务基础地址。

        逻辑规划：
        1. 去除首尾空白，避免请求地址因配置格式错误而失效。
        2. 只允许 HTTP 或 HTTPS 地址，拒绝缺少协议或其他协议的地址。
        3. 移除末尾斜杠，使后续 HTTP 客户端拼接相对路径时行为一致。
        """

        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("NMS_API_BASE_URL must start with http:// or https://")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存当前进程使用的配置实例。

    逻辑规划：
    1. 首次调用时由 Pydantic Settings 读取 config/.env 和环境变量。
    2. 对必填字段和字段类型执行统一校验；校验失败直接抛出异常。
    3. 使用 lru_cache 复用同一实例，避免每次创建 Agent 都重新读取配置。
    """

    return Settings()
