"""消息查询和会话聊天接口。"""

from typing import Any
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import StreamingResponse
from app import database
from app.api.agent import _auth_data_from_payload, _user_id_from_auth_data
from app.api.schemas.messages import ChatRequest, MessageResponse, PendingTaskStateResponse, ResumeRequest
from app.checkpoint import get_checkpointer
from app.context import create_token_counter
from app.memory import compile_thread_context
from app.settings import get_settings
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


@router.get("/{thread_id}/task-state", response_model=PendingTaskStateResponse | None)
async def get_pending_task_state(thread_id: UUID) -> PendingTaskStateResponse | None:
    """读取会话仍待用户处理的人工确认状态。"""

    user_id = _user_id_from_auth_data(_auth_data_from_payload(None))
    try:
        task_state = await to_thread.run_sync(database.load_thread_task_state, thread_id, user_id)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    approval_status = task_state.state.get("approval_status") if task_state is not None else None
    if approval_status not in {"pending", "approved"}:
        return None
    pending_actions = task_state.state.get("pending_actions")
    if not isinstance(pending_actions, list):
        return None
    return PendingTaskStateResponse(
        approval_status=approval_status,
        pending_actions=pending_actions,
    )


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
    try:
        task_state = await to_thread.run_sync(database.load_thread_task_state, thread_id, user_id)
    except RuntimeError as error:
        await release_active_thread(thread_id)
        raise HTTPException(status_code=503, detail=str(error)) from error
    if task_state is not None and task_state.state.get("approval_status") in {"pending", "approved"}:
        await release_active_thread(thread_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Thread has a pending approval; resume it before sending a new message",
        )

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
    try:
        settings = get_settings()
        stored_context_messages = await to_thread.run_sync(
            database.load_stored_messages, thread_id, user_id
        )
        if stored_context_messages is None:
            await release_active_thread(thread_id)
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
        thread_summary = await to_thread.run_sync(
            database.load_thread_summary, thread_id, user_id
        )
        compilation = await compile_thread_context(
            thread_id,
            user_id,
            stored_context_messages,
            thread_summary,
            settings=settings,
            token_counter=create_token_counter(settings),
        )
    except ValueError as error:
        await release_active_thread(thread_id)
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        await release_active_thread(thread_id)
        raise HTTPException(status_code=503, detail=str(error)) from error
    trace_id = str(uuid4())
    return StreamingResponse(
        stream_reply(thread_id, compilation.messages, trace_id, user_id),
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
    user_id = _user_id_from_auth_data(_auth_data_from_payload(None))
    try:
        task_state = await to_thread.run_sync(database.load_thread_task_state, thread_id, user_id)
    except RuntimeError as error:
        await release_active_thread(thread_id)
        raise HTTPException(status_code=503, detail=str(error)) from error
    approval_status = task_state.state.get("approval_status") if task_state is not None else None
    if approval_status not in {"pending", "approved"}:
        await release_active_thread(thread_id)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="No resumable approval for thread")
    if approval_status == "approved" and not request.approved:
        await release_active_thread(thread_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Approval was already granted; resume it with approved=true",
        )
    trace_id = str(uuid4())
    return StreamingResponse(
        stream_reply(
            thread_id,
            [],
            trace_id,
            user_id,
            input_value=resume_command(request.approved, request.reason),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Trace-Id": trace_id},
    )
