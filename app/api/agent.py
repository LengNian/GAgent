"""Agent 用户上下文相关接口。"""

from typing import Any

from fastapi import APIRouter, HTTPException


agent_router = APIRouter(prefix="/api/agent", tags=["agent"])

_DEFAULT_AUTH_DATA: dict[str, Any] = {
    "access_token": "string",
    "expires_in": 0,
    "token_type": "Bearer",
    "user_info": {
        "display_name": "string",
        "granted_permissions": ["string"],
        "mfa_enabled": True,
        "roles": ["string"],
        "tenant_id": "string",
        "user_id": "user-admin-001",
        "username": "admin",
    },
}


def _auth_data_from_payload(payload: object) -> dict[str, Any]:
    """从通用请求对象提取完整认证数据；前端未接入时使用默认模拟数据。"""

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            return dict(data)
    return dict(_DEFAULT_AUTH_DATA)


def _user_id_from_auth_data(auth_data: dict[str, Any]) -> str:
    """从完整认证数据中读取用户 ID。"""

    user_info = auth_data.get("user_info")
    if isinstance(user_info, dict) and isinstance(user_info.get("user_id"), str):
        user_id = user_info["user_id"].strip()
        if user_id:
            return user_id
    return str(_DEFAULT_AUTH_DATA["user_info"]["user_id"])


@agent_router.post("/login")
async def receive_agent_user_context(payload: dict[str, Any]) -> dict[str, Any]:
    """接收前端登录信息并提取用户上下文。"""

    if payload.get("code") != 0:
        message = payload.get("message")
        raise HTTPException(
            status_code=401,
            detail=message if isinstance(message, str) and message else "登录信息无效",
        )

    data = payload.get("data")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Invalid authentication data")

    access_token = data.get("access_token")
    if not isinstance(access_token, str) or not access_token.strip():
        raise HTTPException(status_code=401, detail="Missing access token")

    user_info = data.get("user_info")
    if not isinstance(user_info, dict):
        raise HTTPException(status_code=400, detail="Missing user information")

    return {"code": 0, "message": "success", "data": data}
