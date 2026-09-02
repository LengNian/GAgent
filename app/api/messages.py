"""消息查询和会话聊天接口。"""

from typing import Any
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage

from app import database
from app.api.agent import _auth_data_from_payload, _user_id_from_auth_data
from app.api.schemas.messages import ChatRequest, MessageResponse, ResumeRequest
from app.checkpoint import get_checkpointer
from app.services.chat_service import active_threads, active_threads_lock, release_active_thread, resume_command, stream_reply

router = APIRouter(prefix="/api/threads", tags=["messages"])


@router.get("/{thread_id}/messages", response_model=list[MessageResponse])
async def get_thread_messages(
    thread_id: UUID,
    payload: dict[str, Any] | None = Body(default=None),
) -> list[MessageResponse]:
    """返回指定会话中可供页面展示的用户和助手消息。

    逻辑规划：
    1. 根据 thread_id 和 user_id 从数据库查找消息；不存在或无权访问时返回 404。
    2. 只返回 role 和 content，避免内部消息结构泄露给前端。
    3. 保持原消息列表顺序，返回前端可直接渲染的 role 和 content 字段。
    """

    user_id = _user_id_from_auth_data(_auth_data_from_payload(payload))
    try:
        stored_messages = await to_thread.run_sync(database.load_messages, thread_id, user_id)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if stored_messages is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    return [
        MessageResponse(role=role, content=content) for role, content in stored_messages
    ]


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
    1. 根据 thread_id 和 user_id 从数据库加载历史消息；不存在或无权访问时返回 404。
    2. 检查会话是否正在执行；是则返回 409，避免消息交错。
    3. 标记会话运行中，先追加用户消息，再创建 StreamingResponse。
    4. 具体 Agent 执行和标记清理由 _stream_reply 负责。
    """
    auth_data = _auth_data_from_payload(request.model_dump())
    user_id = _user_id_from_auth_data(auth_data)
    async with active_threads_lock:
        if thread_id in active_threads:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Thread is already running")
        active_threads.add(thread_id)

    try:
        stored_messages = await to_thread.run_sync(database.load_messages, thread_id, user_id)
    except Exception as error:
        await release_active_thread(thread_id)
        if isinstance(error, RuntimeError):
            raise HTTPException(status_code=503, detail=str(error)) from error
        raise
    if stored_messages is None:
        await release_active_thread(thread_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")

    messages = [
        HumanMessage(content=content) if role == "user" else AIMessage(content=content)
        for role, content in stored_messages
    ]
    try:
        persisted = await to_thread.run_sync(
            database.append_message, thread_id, user_id, "user", request.content
        )
    except Exception as error:
        await release_active_thread(thread_id)
        if isinstance(error, RuntimeError):
            raise HTTPException(status_code=503, detail=str(error)) from error
        raise
    if not persisted:
        await release_active_thread(thread_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    if not stored_messages:
        try:
            await to_thread.run_sync(database.set_auto_title_if_empty, thread_id, user_id, request.content)
        except RuntimeError as error:
            await release_active_thread(thread_id)
            raise HTTPException(status_code=503, detail=str(error)) from error
    messages.append(HumanMessage(content=request.content))
    trace_id = str(uuid4())
    return StreamingResponse(
        stream_reply(thread_id, messages, trace_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": trace_id,
        },
    )


@router.post("/{thread_id}/resume")
async def resume_chat(thread_id: UUID, request: ResumeRequest) -> StreamingResponse:
    """恢复当前会话最近一次人工确认中断。"""

    if get_checkpointer() is None:
        raise HTTPException(status_code=503, detail="Checkpoint is not configured")
    async with active_threads_lock:
        if thread_id in active_threads:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Thread is already running")
        active_threads.add(thread_id)
    trace_id = str(uuid4())
    return StreamingResponse(
        stream_reply(
            thread_id,
            [],
            trace_id,
            _user_id_from_auth_data(_auth_data_from_payload(None)),
            input_value=resume_command(request.approved, request.reason),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Trace-Id": trace_id},
    )
