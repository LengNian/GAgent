"""中台"登录换 token"鉴权：token 获取、缓存、续期与 401 重放支撑。

错误处理约定：登录与刷新请求本身不做重试——密码连续错误可能触发中台风控锁定；
凭证问题属于配置错误，应立即失败并提示运维检查，而不是自动重试。
"""

from __future__ import annotations

import asyncio
import logging
import time

import httpx

from mcp_gateway.config_loader import AuthConfig, resolve_env_reference
from mcp_gateway.errors import UpstreamAuthError

logger = logging.getLogger(__name__)

# token 剩余寿命低于该阈值时主动续期，避免请求在临界点 401
_TOKEN_REFRESH_MARGIN_SECONDS = 300


def _dig_field(payload: dict, dotted_path: str) -> str:
    """按 `data.access_token` 形式的点路径从响应中取字符串字段。"""

    node: object = payload
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise UpstreamAuthError(f"中台登录响应缺少字段: {dotted_path}")
        node = node[part]
    if not isinstance(node, str) or not node:
        raise UpstreamAuthError(f"中台登录响应字段不是有效字符串: {dotted_path}")
    return node


class LoginTokenProvider:
    """单个中台的 token 生命周期管理器（非线程安全，按事件循环串行使用）。"""

    def __init__(self, *, base_url: str, auth: AuthConfig) -> None:
        """创建针对一个中台的 token 管理器。

        Args:
            base_url: 中台 base URL（已解析）。
            auth: 该中台的鉴权配置。
        """

        self._base_url = base_url.rstrip("/")
        self._auth = auth
        self._username = resolve_env_reference(auth.username_env, context=f"{auth.username_env}")
        self._password = resolve_env_reference(auth.password_env, context=f"{auth.password_env}")
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: float = 0.0
        self._refresh_lock = asyncio.Lock()

    async def get_valid_token(self, client: httpx.AsyncClient) -> str:
        """返回当前可用 token；过期或临近过期时先续期。

        Args:
            client: 复用的 httpx 客户端。
        Returns:
            有效的 access_token。
        Raises:
            UpstreamAuthError: 登录或刷新最终失败。
        """

        async with self._refresh_lock:
            if self._access_token and not self._is_expiring():
                return self._access_token
            if self._access_token and self._auth.refresh_endpoint and self._refresh_token:
                try:
                    return await self._refresh(client)
                except UpstreamAuthError:
                    logger.warning("gateway_token_refresh_failed_fallback_to_login")
            return await self._login(client)

    async def force_refresh(self, client: httpx.AsyncClient) -> str:
        """401 后强制换新 token（先刷新，失败则重新登录）。

        Args:
            client: 复用的 httpx 客户端。
        Returns:
            新的 access_token。
        Raises:
            UpstreamAuthError: 刷新与登录均失败。
        """

        async with self._refresh_lock:
            if self._auth.refresh_endpoint and self._refresh_token:
                try:
                    return await self._refresh(client)
                except UpstreamAuthError:
                    logger.warning("gateway_token_refresh_failed_fallback_to_login")
            return await self._login(client)

    def _is_expiring(self) -> bool:
        """判断 token 是否已过期或临近过期。"""

        return time.monotonic() >= self._expires_at - _TOKEN_REFRESH_MARGIN_SECONDS

    async def _login(self, client: httpx.AsyncClient) -> str:
        """用账号密码换取新 token 并缓存。"""

        response = await client.post(
            f"{self._base_url}{self._auth.login_endpoint}",
            json={"username": self._username, "password": self._password},
        )
        if response.status_code != 200:
            # 不重试：凭证错误重试只会增加中台风控锁定的风险
            logger.error("gateway_login_failed status=%s", response.status_code)
            raise UpstreamAuthError("中台登录失败，请检查网关凭证配置。")
        payload = response.json()
        self._access_token = _dig_field(payload, self._auth.token_field)
        if self._auth.refresh_token_field:
            try:
                self._refresh_token = _dig_field(payload, self._auth.refresh_token_field)
            except UpstreamAuthError:
                self._refresh_token = None
        expires_in = 86400
        if self._auth.expires_in_field:
            try:
                expires_in = int(_dig_field(payload, self._auth.expires_in_field))
            except (UpstreamAuthError, ValueError):
                expires_in = 86400
        self._expires_at = time.monotonic() + expires_in
        logger.info("gateway_login_succeeded expires_in_seconds=%s", expires_in)
        return self._access_token

    async def _refresh(self, client: httpx.AsyncClient) -> str:
        """用 refresh_token 换新 access_token。"""

        assert self._auth.refresh_endpoint is not None
        response = await client.post(
            f"{self._base_url}{self._auth.refresh_endpoint}",
            json={"refresh_token": self._refresh_token},
        )
        if response.status_code != 200:
            raise UpstreamAuthError(f"中台 token 刷新失败: HTTP {response.status_code}")
        payload = response.json()
        new_token = _dig_field(payload, self._auth.token_field)
        self._access_token = new_token
        expires_in = 86400
        if self._auth.expires_in_field:
            try:
                expires_in = int(_dig_field(payload, self._auth.expires_in_field))
            except (UpstreamAuthError, ValueError):
                expires_in = 86400
        self._expires_at = time.monotonic() + expires_in
        logger.info("gateway_token_refreshed expires_in_seconds=%s", expires_in)
        return new_token
