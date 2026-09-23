"""从环境文件加载并校验应用配置。"""

from functools import lru_cache

from typing import Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
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
    # DeepSeek 等兼容供应商的思考模式会拒绝具名 tool_choice、要求回传
    # reasoning_content，并静默忽略 temperature，因此提供显式关闭开关。
    # 默认不开启，保证对不需要该参数的供应商零影响。
    llm_disable_thinking: bool = Field(default=False, validation_alias="LLM_DISABLE_THINKING")
    mcp_gateway_url: str = Field(validation_alias="MCP_GATEWAY_URL")
    mcp_gateway_token: SecretStr = Field(validation_alias="MCP_GATEWAY_TOKEN")
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL")
    stepfun_api_key: SecretStr | None = Field(default=None, validation_alias="STEPFUN_API_KEY")
    stepfun_base_url: str = Field(
        default="https://api.stepfun.com/v1", validation_alias="STEPFUN_BASE_URL"
    )
    stepfun_asr_model: str = Field(
        default="stepaudio-2.5-asr", validation_alias="STEPFUN_ASR_MODEL", min_length=1
    )
    stepfun_tts_model: str = Field(
        default="step-tts-mini", validation_alias="STEPFUN_TTS_MODEL", min_length=1
    )
    stepfun_tts_voice: str = Field(
        default="cixingnansheng", validation_alias="STEPFUN_TTS_VOICE", min_length=1
    )
    stepfun_tts_language: Literal["粤语", "四川话", "日语"] | None = Field(
        default=None, validation_alias="STEPFUN_TTS_LANGUAGE"
    )
    stepfun_tts_emotion: str | None = Field(
        default=None, validation_alias="STEPFUN_TTS_EMOTION", min_length=1
    )
    stepfun_audio_timeout_seconds: float = Field(
        default=60, validation_alias="STEPFUN_AUDIO_TIMEOUT_SECONDS", gt=0
    )
    stepfun_asr_max_bytes: int = Field(
        default=10 * 1024 * 1024, validation_alias="STEPFUN_ASR_MAX_BYTES", ge=1
    )
    context_max_tokens: int = Field(default=12000, validation_alias="CONTEXT_MAX_TOKENS", ge=1)
    context_max_message_tokens: int = Field(
        default=3000, validation_alias="CONTEXT_MAX_MESSAGE_TOKENS", ge=1
    )
    context_summary_max_tokens: int = Field(
        default=1200, validation_alias="CONTEXT_SUMMARY_MAX_TOKENS", ge=1
    )
    context_compaction_trigger_ratio: float = Field(
        default=0.8, validation_alias="CONTEXT_COMPACTION_TRIGGER_RATIO", gt=0, le=1
    )
    context_compaction_target_ratio: float = Field(
        default=0.4, validation_alias="CONTEXT_COMPACTION_TARGET_RATIO", gt=0, lt=1
    )
    context_min_recent_rounds: int = Field(
        default=10, validation_alias="CONTEXT_MIN_RECENT_ROUNDS", ge=0
    )
    long_term_memory_enabled: bool = Field(
        default=True, validation_alias="LONG_TERM_MEMORY_ENABLED"
    )
    long_term_memory_extraction_model: str | None = Field(
        default=None, validation_alias="LONG_TERM_MEMORY_EXTRACTION_MODEL"
    )
    long_term_memory_extraction_temperature: float = Field(
        default=0, validation_alias="LONG_TERM_MEMORY_EXTRACTION_TEMPERATURE", ge=0, le=2
    )
    long_term_memory_extraction_max_tokens: int = Field(
        default=600, validation_alias="LONG_TERM_MEMORY_EXTRACTION_MAX_TOKENS", ge=1
    )
    long_term_memory_extraction_timeout_seconds: float = Field(
        default=30, validation_alias="LONG_TERM_MEMORY_EXTRACTION_TIMEOUT_SECONDS", gt=0
    )
    long_term_memory_max_candidates: int = Field(
        default=5, validation_alias="LONG_TERM_MEMORY_MAX_CANDIDATES", ge=1
    )
    long_term_memory_max_content_length: int = Field(
        default=240, validation_alias="LONG_TERM_MEMORY_MAX_CONTENT_LENGTH", ge=1
    )
    long_term_memory_min_importance: int = Field(
        default=4, validation_alias="LONG_TERM_MEMORY_MIN_IMPORTANCE", ge=0, le=10
    )
    long_term_memory_max_evidence_messages: int = Field(
        default=4, validation_alias="LONG_TERM_MEMORY_MAX_EVIDENCE_MESSAGES", ge=1
    )
    long_term_memory_embedding_enabled: bool = Field(
        default=True, validation_alias="LONG_TERM_MEMORY_EMBEDDING_ENABLED"
    )
    long_term_memory_embedding_model: str = Field(
        default="BAAI/bge-base-zh-v1.5",
        validation_alias="LONG_TERM_MEMORY_EMBEDDING_MODEL",
        min_length=1,
    )
    long_term_memory_embedding_dimensions: int = Field(
        default=768,
        validation_alias="LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS",
        ge=768,
        le=2048,
    )
    long_term_memory_embedding_device: str = Field(
        default="cpu", validation_alias="LONG_TERM_MEMORY_EMBEDDING_DEVICE", min_length=1
    )
    long_term_memory_embedding_timeout_seconds: float = Field(
        default=30,
        validation_alias="LONG_TERM_MEMORY_EMBEDDING_TIMEOUT_SECONDS",
        gt=0,
    )
    long_term_memory_semantic_dedup_enabled: bool = Field(
        default=True, validation_alias="LONG_TERM_MEMORY_SEMANTIC_DEDUP_ENABLED"
    )
    long_term_memory_semantic_dedup_distance: float = Field(
        default=0.08,
        validation_alias="LONG_TERM_MEMORY_SEMANTIC_DEDUP_DISTANCE",
        ge=0,
        le=2,
    )
    long_term_memory_conflict_resolution_enabled: bool = Field(
        default=True,
        validation_alias="LONG_TERM_MEMORY_CONFLICT_RESOLUTION_ENABLED",
    )
    long_term_memory_conflict_candidate_limit: int = Field(
        default=10,
        validation_alias="LONG_TERM_MEMORY_CONFLICT_CANDIDATE_LIMIT",
        ge=1,
    )
    long_term_memory_recall_enabled: bool = Field(
        default=True, validation_alias="LONG_TERM_MEMORY_RECALL_ENABLED"
    )
    long_term_memory_recall_candidate_limit: int = Field(
        default=10, validation_alias="LONG_TERM_MEMORY_RECALL_CANDIDATE_LIMIT", ge=1
    )
    long_term_memory_recall_top_n: int = Field(
        default=3, validation_alias="LONG_TERM_MEMORY_RECALL_TOP_N", ge=1
    )
    long_term_memory_recall_min_score: float = Field(
        default=0.5, validation_alias="LONG_TERM_MEMORY_RECALL_MIN_SCORE", ge=0, le=1
    )
    long_term_memory_recall_relevance_weight: float = Field(
        default=0.7, validation_alias="LONG_TERM_MEMORY_RECALL_RELEVANCE_WEIGHT", ge=0, le=1
    )
    long_term_memory_recall_importance_weight: float = Field(
        default=0.2, validation_alias="LONG_TERM_MEMORY_RECALL_IMPORTANCE_WEIGHT", ge=0, le=1
    )
    long_term_memory_recall_recency_weight: float = Field(
        default=0.1, validation_alias="LONG_TERM_MEMORY_RECALL_RECENCY_WEIGHT", ge=0, le=1
    )
    long_term_memory_recall_profile_decay_k: float = Field(
        default=0.001, validation_alias="LONG_TERM_MEMORY_RECALL_PROFILE_DECAY_K", ge=0
    )
    long_term_memory_recall_preference_decay_k: float = Field(
        default=0.003, validation_alias="LONG_TERM_MEMORY_RECALL_PREFERENCE_DECAY_K", ge=0
    )
    long_term_memory_recall_commitment_decay_k: float = Field(
        default=0.01, validation_alias="LONG_TERM_MEMORY_RECALL_COMMITMENT_DECAY_K", ge=0
    )
    long_term_memory_injection_enabled: bool = Field(
        default=True, validation_alias="LONG_TERM_MEMORY_INJECTION_ENABLED"
    )
    long_term_memory_injection_max_tokens: int = Field(
        default=512, validation_alias="LONG_TERM_MEMORY_INJECTION_MAX_TOKENS", ge=1
    )
    llm_tokenizer_backend: Literal["huggingface"] = Field(
        default="huggingface", validation_alias="LLM_TOKENIZER_BACKEND"
    )
    llm_tokenizer_name: str = Field(
        default="zai-org/GLM-4.5-Air", validation_alias="LLM_TOKENIZER_NAME", min_length=1
    )
    llm_tokenizer_revision: str | None = Field(
        default=None, validation_alias="LLM_TOKENIZER_REVISION"
    )
    llm_tokenizer_trust_remote_code: bool = Field(
        default=True, validation_alias="LLM_TOKENIZER_TRUST_REMOTE_CODE"
    )

    @field_validator("mcp_gateway_url")
    @classmethod
    def mcp_gateway_url_must_use_http(cls, value: str) -> str:
        """校验 MCP 网关地址。

        逻辑规划：
        1. 去除首尾空白，避免请求地址因配置格式错误而失效。
        2. 只允许 HTTP 或 HTTPS 地址，拒绝缺少协议或其他协议的地址。
        3. 移除末尾斜杠，使客户端拼接 MCP 路径时行为一致。
        """

        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("MCP_GATEWAY_URL must start with http:// or https://")
        return value

    @model_validator(mode="after")
    def summary_budget_must_fit_context_budget(self) -> "Settings":
        """确保摘要预算不会耗尽当前问题的上下文空间。"""

        if self.context_summary_max_tokens >= self.context_max_tokens:
            raise ValueError("CONTEXT_SUMMARY_MAX_TOKENS must be smaller than CONTEXT_MAX_TOKENS")
        if self.context_compaction_target_ratio >= self.context_compaction_trigger_ratio:
            raise ValueError(
                "CONTEXT_COMPACTION_TARGET_RATIO must be smaller than "
                "CONTEXT_COMPACTION_TRIGGER_RATIO"
            )
        recall_weight_total = (
            self.long_term_memory_recall_relevance_weight
            + self.long_term_memory_recall_importance_weight
            + self.long_term_memory_recall_recency_weight
        )
        if abs(recall_weight_total - 1) > 1e-9:
            raise ValueError("Long-term memory recall weights must sum to 1")
        if self.long_term_memory_recall_top_n > self.long_term_memory_recall_candidate_limit:
            raise ValueError(
                "LONG_TERM_MEMORY_RECALL_TOP_N must not exceed "
                "LONG_TERM_MEMORY_RECALL_CANDIDATE_LIMIT"
            )
        if self.stepfun_tts_language and self.stepfun_tts_emotion:
            raise ValueError("STEPFUN_TTS_LANGUAGE and STEPFUN_TTS_EMOTION cannot both be set")
        return self

    def llm_extra_body(self) -> dict[str, Any] | None:
        """返回聊天模型请求的供应商私有参数；未启用时返回 None。

        Returns:
            开启 LLM_DISABLE_THINKING 时返回思考模式关闭参数，否则返回 None。
            注意只有 `thinking.type=disabled` 能真正关闭思考；
            `reasoning_effort=minimal` 会被映射为 low，仍处于思考模式。
        """

        if not self.llm_disable_thinking:
            return None
        return {"thinking": {"type": "disabled"}}


# @lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存当前进程使用的配置实例。

    逻辑规划：
    1. 首次调用时由 Pydantic Settings 读取 config/.env 和环境变量。
    2. 对必填字段和字段类型执行统一校验；校验失败直接抛出异常。
    3. 使用 lru_cache 复用同一实例，避免每次创建 Agent 都重新读取配置。
    """

    return Settings()
