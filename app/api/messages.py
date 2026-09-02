"""消息查询和会话聊天接口。"""

from fastapi import APIRouter
from app.api.schemas.messages import MessageResponse
from app.api.threads import get_thread_messages as _get_thread_messages
from app.api.threads import stream_chat as _stream_chat


router = APIRouter(prefix="/api/threads", tags=["messages"])
router.add_api_route(
    "/{thread_id}/messages",
    _get_thread_messages,
    methods=["GET"],
    response_model=list[MessageResponse],
)
router.add_api_route(
    "/{thread_id}/chat",
    _stream_chat,
    methods=["POST"],
)
