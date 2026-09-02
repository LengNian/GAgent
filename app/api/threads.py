"""创建会话和流式返回 Agent 回复的 HTTP 接口。"""

import asyncio
import json
import logging
from typing import Any
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import APIRouter, Body, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from app.api.agent import _DEFAULT_AUTH_DATA, _auth_data_from_payload, _user_id_from_auth_data
from app.api.schemas.messages import ChatRequest, MessageResponse
from app.api.schemas.threads import ThreadCreatedResponse, ThreadSummaryResponse, ThreadTitleRequest

from app.agent import create_agent
from app.agent.factory import AgentExecutionLimitError
from app.agent_manifest import get_agents_config
from app import database
from app.observability import log_event, reset_trace_id, set_trace_id


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/threads", tags=["threads"])
_active_threads: set[UUID] = set()
_active_threads_lock = asyncio.Lock()


async def _release_active_thread(thread_id: UUID) -> None:
    """在并发锁保护下释放会话运行标记。"""

    async with _active_threads_lock:
        _active_threads.discard(thread_id)


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


def _report_output_text(output: object) -> str:
    """从 Report Node 的状态输出中提取最终助手文本。

    逻辑规划：
    1. 只接受包含 messages 列表的节点状态输出。
    2. 读取最后一条消息，因为 Report Node 只追加一条最终报告消息。
    3. 复用消息文本提取逻辑；未知结构返回空文本，不发送伪造 SSE 内容。
    """

    if not isinstance(output, dict):
        return ""
    messages = output.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return _chunk_text(messages[-1])


def _safe_tool_arguments(arguments: object) -> dict[str, object]:
    """整理可展示的工具参数，避免把敏感值或复杂原始结构发送到前端。

    逻辑规划：
    1. 只接受字典参数；模型返回的其他结构视为不可展示，返回空字典。
    2. 对名称包含 token、key、password、secret 或 credential 的字段隐藏值，避免敏感信息进入 SSE。
    3. 仅保留 JSON 基础类型；复杂对象不展开，防止前端误把内部工具数据当作用户可读结果。
    4. 参数整理失败时保持空结果，不影响 Agent 继续执行工具调用。
    """

    if not isinstance(arguments, dict):
        return {}

    sensitive_words = ("token", "key", "password", "secret", "credential")
    safe_arguments: dict[str, object] = {}
    for name, value in arguments.items():
        field_name = str(name)
        if any(word in field_name.lower() for word in sensitive_words):
            safe_arguments[field_name] = "[已隐藏]"
        elif value is None or isinstance(value, (str, int, float, bool)):
            safe_arguments[field_name] = value
        else:
            safe_arguments[field_name] = "[复杂参数]"
    return safe_arguments


def _tool_agent_name(tool_name: str) -> str:
    """根据 Action allowlist 返回工具所属 Agent 的展示名称。"""

    display_names = {
        "conversation_agent": "Conversation Agent",
        "iot_agent": "IoT Agent",
    }
    for manifest in get_agents_config().agents:
        if tool_name in manifest.allowed_actions:
            return display_names.get(manifest.agent_id, manifest.agent_id)
    return "领域 Agent"


def _agent_display_name(agent_id: str) -> str:
    """返回可展示的 Agent 名称，避免直接展示内部标识。"""

    display_names = {
        "conversation_agent": "Conversation Agent",
        "iot_agent": "IoT Agent",
    }
    return display_names.get(agent_id, agent_id)


async def _stream_reply(
    thread_id: UUID,
    messages: list[BaseMessage],
    trace_id: str,
    user_id: str | None = None,
):
    """执行 Agent 并将回复转换为 SSE 流。

    Args:
        thread_id: 当前正在执行的会话 ID。
        user_id: 当前会话所属用户 ID；旧的内部调用可不传入。
        messages: 传给 Agent 的当前会话消息列表。
    Yields:
        文本增量、完成或失败事件对应的 SSE 字符串。
    逻辑规划：
    1. 创建 Agent，并监听 LangGraph v2 的节点和工具生命周期事件。
    2. Supervisor 节点结束时，读取模型实际返回的任务摘要和路由结果并转换为进度事件。
    3. 处理模型流式文本；模型结束事件仅在没有文本流时补发文本，避免最终回答重复。
    4. 工具参数只发送脱敏后的基础类型，不发送隐藏思维链或原始工具 JSON。
    5. 正常完成后追加助手消息并发送 done；任意执行异常转换为统一 error 事件。
    6. 无论成功或失败都释放会话运行标记，避免会话永久处于执行状态。
    """
    trace_token = set_trace_id(trace_id)
    assistant_text = ""
    started_tool_runs: set[str] = set()
    completed_tool_runs: set[str] = set()
    streamed_model_runs: set[str] = set()

    try:
        log_event(logger, logging.INFO, "agent_request_started", thread_id=str(thread_id))
        agent = create_agent()
        async for event in agent.astream_events({"messages": messages}, version="v2"):

            event_name = event.get("event")
            event_data = event.get("data") or {}
            run_id = str(event.get("run_id") or "")
            node_name = (event.get("metadata") or {}).get("langgraph_node")

            if event_name == "on_chain_end" and node_name == "supervisor":
                output = event_data.get("output")
                if not isinstance(output, dict):
                    continue
                decision_summary = output.get("decision_summary")
                target_agent = output.get("target_agent")

                if isinstance(decision_summary, str) and decision_summary:
                    log_event(
                        logger,
                        logging.INFO,
                        "supervisor_route_decided",
                        thread_id=str(thread_id),
                        target_agent=target_agent,
                    )
                    yield _format_sse_event(
                        "agent_progress",
                        {"message": f"Supervisor：{decision_summary}"},
                    )
                if target_agent in {"conversation_agent", "iot_agent"}:
                    yield _format_sse_event(
                        "agent_progress",
                        {
                            "message": (
                                "Supervisor：我将使用 "
                                f"{_agent_display_name(target_agent)} 处理。"
                            )
                        },
                    )
                continue

            if event_name == "on_chain_start" and node_name == "conversation_agent":
                yield _format_sse_event(
                    "agent_progress",
                    {"message": "Conversation Agent：正在生成回复。"},
                )
                continue

            if event_name == "on_chain_start" and node_name == "iot_agent":
                yield _format_sse_event(
                    "agent_progress",
                    {"message": "IoT Agent：正在核对设备 IP。"},
                )
                continue

            if event_name == "on_chain_start" and node_name == "report":
                yield _format_sse_event(
                    "agent_progress",
                    {"message": "IoT Agent：正在生成查询报告。"},
                )
                continue

            if event_name == "on_chain_end" and node_name == "report":
                text = _report_output_text(event_data.get("output"))
                if text:
                    assistant_text += text
                    yield _format_sse_event("delta", {"text": text})
                continue

            if event_name == "on_tool_start":
                if run_id in started_tool_runs:
                    continue
                started_tool_runs.add(run_id)
                tool_name = str(event.get("name") or "工具")
                tool_input = event_data.get("input", {})
                agent_name = _tool_agent_name(tool_name)
                log_event(
                    logger,
                    logging.INFO,
                    "tool_started",
                    thread_id=str(thread_id),
                    tool_name=tool_name,
                    agent_name=agent_name,
                    arguments=_safe_tool_arguments(tool_input),
                )
                yield _format_sse_event(
                    "tool_start",
                    {
                        "tool_name": tool_name,
                        "arguments": _safe_tool_arguments(tool_input),
                        "message": f"{agent_name}：调用 {tool_name}。",
                    },
                )
                continue

            if event_name == "on_tool_end":
                if run_id in completed_tool_runs:
                    continue
                completed_tool_runs.add(run_id)
                tool_name = str(event.get("name") or "工具")
                agent_name = _tool_agent_name(tool_name)
                log_event(
                    logger,
                    logging.INFO,
                    "tool_finished",
                    thread_id=str(thread_id),
                    tool_name=tool_name,
                    agent_name=agent_name,
                )
                yield _format_sse_event(
                    "tool_end",
                    {
                        "tool_name": tool_name,
                        "message": f"{agent_name}：工具结果已返回，正在生成查询报告。",
                    },
                )
                continue

            if event_name == "on_chat_model_stream":
                if node_name == "report":
                    continue
                text = _chunk_text(event_data.get("chunk"))
                if not text:
                    continue
                streamed_model_runs.add(run_id)
                assistant_text += text
                yield _format_sse_event("delta", {"text": text})
                continue

            if event_name == "on_chat_model_end" and run_id not in streamed_model_runs:
                if node_name == "report":
                    continue
                text = _chunk_text(event_data.get("output"))
                if text and not getattr(event_data.get("output"), "tool_calls", None):
                    assistant_text += text
                    yield _format_sse_event("delta", {"text": text})

        messages.append(AIMessage(content=assistant_text))
        if user_id is not None:
            await to_thread.run_sync(
                database.append_message, thread_id, user_id, "assistant", assistant_text
            )
            
        log_event(
            logger,
            logging.INFO,
            "agent_request_completed",
            thread_id=str(thread_id),
            response_length=len(assistant_text),
        )
        yield _format_sse_event(
            "done",
            {
                "trace_id": trace_id,
                "message": {"role": "assistant", "content": assistant_text},
            },
        )
    except AgentExecutionLimitError as error:
        log_event(
            logger,
            logging.WARNING,
            "agent_execution_limit_reached",
            thread_id=str(thread_id),
            agent_id=error.agent_id,
            limit_type=error.limit_type,
            limit_value=error.limit_value,
        )
        if error.limit_type == "timeout_seconds":
            message = f"{error.agent_id} 执行超时，已安全停止，请稍后重试。"
        else:
            message = f"{error.agent_id} 执行步骤超过限制，已安全停止。"
        yield _format_sse_event(
            "error",
            {"code": "agent_execution_limit", "message": message, "trace_id": trace_id},
        )
    except Exception:
        log_event(logger, logging.ERROR, "agent_request_failed", thread_id=str(thread_id))
        logger.exception("Agent execution failed for thread %s", thread_id)
        yield _format_sse_event(
            "error",
            {
                "code": "agent_execution_failed",
                "message": "Agent 执行失败，请稍后重试。",
                "trace_id": trace_id,
            },
        )
    finally:
        await _release_active_thread(thread_id)
        reset_trace_id(trace_token)


@router.post("", response_model=ThreadCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_thread(payload: dict[str, Any] | None = Body(default=None)) -> ThreadCreatedResponse:
    """创建并返回服务端生成的新会话 ID。

    Returns:
        新进程内会话对应的 UUID。
    逻辑规划：
    1. 使用服务端 uuid4 生成不可由前端指定的会话 ID。
    2. 为该 ID 初始化空消息列表。
    3. 返回标准化响应，不在创建接口隐式添加消息。
    """

    thread_id = uuid4()
    user_id = _user_id_from_auth_data(_auth_data_from_payload(payload))
    try:
        await to_thread.run_sync(database.create_thread, thread_id, user_id)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return ThreadCreatedResponse(thread_id=thread_id)


@router.get("", response_model=list[ThreadSummaryResponse])
async def list_user_threads(payload: dict[str, Any] | None = Body(default=None)):
    """返回当前用户的会话列表。"""
    user_id = _user_id_from_auth_data(_auth_data_from_payload(payload))
    try:
        rows = await to_thread.run_sync(database.list_threads, user_id)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return [ThreadSummaryResponse(thread_id=row[0], title=row[1], title_is_custom=row[2], updated_at=row[3].isoformat()) for row in rows]


@router.patch("/{thread_id}")
async def update_thread_title(thread_id: UUID, request: ThreadTitleRequest):
    """设置或清除当前用户会话标题。"""
    user_id = _user_id_from_auth_data(_DEFAULT_AUTH_DATA)
    try:
        updated = await to_thread.run_sync(database.update_thread_title, thread_id, user_id, request.title)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if not updated:
        raise HTTPException(status_code=404, detail="Thread not found")
    return {"thread_id": thread_id, "title": request.title, "title_is_custom": request.title is not None}


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user_thread(thread_id: UUID) -> None:
    """删除当前用户拥有的会话及其数据库级联关联数据。"""
    user_id = _user_id_from_auth_data(_DEFAULT_AUTH_DATA)
    try:
        deleted = await to_thread.run_sync(database.delete_thread, thread_id, user_id)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    if not deleted:
        raise HTTPException(status_code=404, detail="Thread not found")


# Message routes are registered by app.api.messages.
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


# Chat route is registered by app.api.messages.
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
    async with _active_threads_lock:
        if thread_id in _active_threads:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Thread is already running")
        _active_threads.add(thread_id)

    try:
        stored_messages = await to_thread.run_sync(database.load_messages, thread_id, user_id)
    except Exception as error:
        await _release_active_thread(thread_id)
        if isinstance(error, RuntimeError):
            raise HTTPException(status_code=503, detail=str(error)) from error
        raise
    if stored_messages is None:
        await _release_active_thread(thread_id)
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
        await _release_active_thread(thread_id)
        if isinstance(error, RuntimeError):
            raise HTTPException(status_code=503, detail=str(error)) from error
        raise
    if not persisted:
        await _release_active_thread(thread_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Thread not found")
    if not stored_messages:
        try:
            await to_thread.run_sync(database.set_auto_title_if_empty, thread_id, user_id, request.content)
        except RuntimeError as error:
            await _release_active_thread(thread_id)
            raise HTTPException(status_code=503, detail=str(error)) from error
    messages.append(HumanMessage(content=request.content))
    trace_id = str(uuid4())
    return StreamingResponse(
        _stream_reply(thread_id, messages, trace_id, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Trace-Id": trace_id,
        },
    )
