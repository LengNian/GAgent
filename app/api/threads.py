"""创建会话和流式返回 Agent 回复的 HTTP 接口。"""

import json
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator

from app.agent import create_agent


class ThreadCreatedResponse(BaseModel):
    """创建会话后返回的响应模型。"""

    thread_id: UUID = Field(description="Server-generated UUIDv4 thread identifier")


class ChatRequest(BaseModel):
    """聊天接口接收的用户输入模型。"""

    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        """去除首尾空白，并拒绝不包含有效内容的消息。

        逻辑规划：
        1. 去除用户输入首尾的空白字符，统一后续保存和发送的内容。
        2. 如果清理后为空，立即抛出校验异常，由 FastAPI 返回 422。
        3. 返回清理后的文本，避免空白差异进入 Agent 上下文。
        """

        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


router = APIRouter(prefix="/api/threads", tags=["threads"])
_thread_messages: dict[UUID, list[BaseMessage]] = {}
_active_threads: set[UUID] = set()



def _format_sse_event(event: str, data: dict[str, object]) -> str:
    """将事件名称和 JSON 数据格式化为一条 SSE 消息。

    逻辑规划：
    1. 将事件名称写入 event 行。
    2. 将结构化数据序列化为不转义中文的 JSON。
    3. 使用两个换行符结束事件，保证浏览器可以识别一个完整 SSE 消息。
    """

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(chunk: object) -> str:
    """从模型流式消息中提取可展示的文本。

    逻辑规划：
    1. 读取流式消息对象的 content 属性，不假设具体消息类型。
    2. content 是字符串时直接返回。
    3. content 是内容块列表时提取每个文本块并按原顺序拼接。
    4. 遇到未知结构或没有文本时返回空字符串，让上层跳过该事件。
    """

    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return ""


async def _stream_reply(thread_id: UUID, messages: list[BaseMessage]):
    """执行 Agent 并将回复转换为 SSE 流。

    Args:
        thread_id: 当前正在执行的会话 ID。
        messages: 传给 Agent 的进程内消息列表。
    Yields:
        文本增量、完成或失败事件对应的 SSE 字符串。
    逻辑规划：
    1. 创建 Agent 并读取模型消息流。
    2. 提取文本片段，发送 delta 事件并累计最终助手文本。
    3. 正常完成后追加助手消息，再发送 done 事件。
    4. 任意执行异常转换为统一 error 事件，最后释放会话运行标记。
    """
    assistant_text = ""

    try:
        agent = create_agent()
        async for event in agent.astream({"messages": messages}, stream_mode="messages"):
            message = event[0] if isinstance(event, tuple) else event
            text = _chunk_text(message)
            if not text:
                continue
            assistant_text += text
            yield _format_sse_event("delta", {"text": text})

        messages.append(AIMessage(content=assistant_text))
        yield _format_sse_event("done", {"message": {"role": "assistant", "content": assistant_text}})
    except Exception:
        yield _format_sse_event(
            "error",
            {"code": "agent_execution_failed", "message": "Agent 执行失败，请稍后重试。"},
        )
    finally:
        _active_threads.discard(thread_id)


@router.post("", response_model=ThreadCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_thread() -> ThreadCreatedResponse:
    """创建并返回服务端生成的新会话 ID。

    Returns:
        新进程内会话对应的 UUID。
    逻辑规划：
    1. 使用服务端 uuid4 生成不可由前端指定的会话 ID。
    2. 为该 ID 初始化空消息列表。
    3. 返回标准化响应，不在创建接口隐式添加消息。
    """

    thread_id = uuid4()
    _thread_messages[thread_id] = []
    return ThreadCreatedResponse(thread_id=thread_id)


@router.post("/{thread_id}/chat")
async def stream_chat(thread_id: UUID, request: ChatRequest) -> StreamingResponse:
    """为已有会话执行 Agent，并以 SSE 流返回回复。

    Args:
        thread_id: 要继续的会话 UUID。
        request: 已通过 Pydantic 校验的用户消息。
    Returns:
        包含 Agent 输出事件的 SSE 响应。
    Raises:
        HTTPException: 会话不存在或同一会话已有请求执行时抛出。
    逻辑规划：
    1. 根据 thread_id 查找消息列表；不存在时返回 404。
    2. 检查会话是否正在执行；是则返回 409，避免消息交错。
    3. 标记会话运行中，先追加用户消息，再创建 StreamingResponse。
    4. 具体 Agent 执行和标记清理由 _stream_reply 负责。
    """

    messages = _thread_messages.get(thread_id)
    if messages is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    if thread_id in _active_threads:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Thread is already running")

    _active_threads.add(thread_id)
    messages.append(HumanMessage(content=request.content))
    return StreamingResponse(
        _stream_reply(thread_id, messages),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
