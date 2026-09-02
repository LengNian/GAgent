"""创建会话和流式返回 Agent 回复的 HTTP 接口。"""

import logging
from typing import Any
from uuid import UUID, uuid4

from anyio import to_thread
from fastapi import APIRouter, Body, HTTPException, status
from app.api.agent import _DEFAULT_AUTH_DATA, _auth_data_from_payload, _user_id_from_auth_data
from app.api.schemas.threads import ThreadCreatedResponse, ThreadSummaryResponse, ThreadTitleRequest
from app import database


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/threads", tags=["threads"])


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
