"""会话 API 的请求与响应模型。"""

from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class ThreadCreatedResponse(BaseModel):
    """创建会话后返回的响应模型。"""

    thread_id: UUID = Field(description="Server-generated UUIDv4 thread identifier")


class ThreadSummaryResponse(BaseModel):
    """会话列表中的摘要信息。"""

    thread_id: UUID
    title: str | None
    title_is_custom: bool
    updated_at: str


class ThreadTitleRequest(BaseModel):
    """设置或清除会话标题的请求模型。"""

    title: str | None = Field(default=None, max_length=80)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None
