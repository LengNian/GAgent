"""In-memory thread endpoints."""

import json
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator

from app.agent import create_agent


class ThreadCreatedResponse(BaseModel):
    """Response returned after creating a new thread."""

    thread_id: UUID = Field(description="Server-generated UUIDv4 thread identifier")


class ChatRequest(BaseModel):
    """User input accepted by the chat endpoint."""

    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content must not be blank")
        return value


router = APIRouter(prefix="/api/threads", tags=["threads"])
_thread_messages: dict[UUID, list[BaseMessage]] = {}
_active_threads: set[UUID] = set()


def _sse(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _chunk_text(chunk: object) -> str:
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
    assistant_text = ""

    try:
        agent = create_agent()
        async for event in agent.astream({"messages": messages}, stream_mode="messages"):
            message = event[0] if isinstance(event, tuple) else event
            text = _chunk_text(message)
            if not text:
                continue
            assistant_text += text
            yield _sse("delta", {"text": text})

        messages.append(AIMessage(content=assistant_text))
        yield _sse("done", {"message": {"role": "assistant", "content": assistant_text}})
    except Exception:
        yield _sse(
            "error",
            {"code": "agent_execution_failed", "message": "Agent 执行失败，请稍后重试。"},
        )
    finally:
        _active_threads.discard(thread_id)


@router.post("", response_model=ThreadCreatedResponse, status_code=status.HTTP_201_CREATED)
async def create_thread() -> ThreadCreatedResponse:
    """Create a thread identifier without persisting it."""

    thread_id = uuid4()
    _thread_messages[thread_id] = []
    return ThreadCreatedResponse(thread_id=thread_id)


@router.post("/{thread_id}/chat")
async def chat(thread_id: UUID, request: ChatRequest) -> StreamingResponse:
    """Stream one Agent response for an in-memory thread."""

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
