"""消息 API 的请求与响应模型。"""

from pydantic import BaseModel, field_validator


class MessageResponse(BaseModel):
    """供前端恢复展示的业务消息响应模型。"""

    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天接口接收的用户输入模型。"""

    content: str

    model_config = {"extra": "allow"}

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        """去除首尾空白，并拒绝不包含有效内容的消息。"""

        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value
