"""把 MCP 工具调用翻译成对中台的 HTTP 请求并执行。

泛化自 app/tools/http_client.py 的既有逻辑（参数路由、指数退避、错误翻译），
并增加：登录换 token 注入、BaseResp 业务码判定、401 刷新重放。
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import httpx

from mcp_gateway.auth import LoginTokenProvider
from mcp_gateway.config_loader import ApiConfig, PlatformConfig
from mcp_gateway.errors import ToolConfigurationError, ToolExecutionError

logger = logging.getLogger(__name__)

_SUCCESS_CODE = 0
# 401 属于凭证问题：重放一次（内部刷新 token），不计入常规重试次数
_MAX_AUTH_REPLAYS = 1


def _format_number(value: float) -> str:
    """数值转字符串时避免科学计数法，保证中台 query/path 参数可读。"""

    return format(value, ".12g")


def _stringify_argument(value: Any) -> str:
    """把工具参数转为 HTTP 参数字符串（bool 转小写，符合常见 REST 约定）。"""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


async def execute_api_call(
    *,
    platform: PlatformConfig,
    api: ApiConfig,
    base_url: str,
    token_provider: LoginTokenProvider,
    client: httpx.AsyncClient,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """执行一次 API 调用并返回中台业务数据。

    逻辑规划：
    1. [参数路由] 按 argument_locations 把参数分发到 path、query、body。
       - 约束: path 参数缺失属于网关配置错误（schema 已保证必填，此处防御 path 占位符）。
    2. [鉴权] 从 token_provider 取 token，注入 Authorization: Bearer。
    3. [执行] 按配置做指数退避重试；网络错误与 5xx 可重试，400/404 不重试。
    4. [401 处理] token 失效时强制刷新并重放一次，重放仍 401 才报错。
    5. [业务判定] HTTP 200 后检查 BaseResp.code：非 0 视为业务失败。
       - 原因: 中台所有接口统一 BaseResp（code=0 成功），业务失败重试无意义。
    6. [返回] 仅返回 data 字段；异常分支全部转成 ToolExecutionError 安全文案。

    Args:
        platform: 目标中台配置。
        api: 工具对应的 API 映射。
        base_url: 中台 base URL（已解析）。
        token_provider: 该中台的 token 管理器。
        client: 复用的 httpx 客户端。
        arguments: 已经过 schema 校验的工具参数。
    Returns:
        中台响应中的 data 字段。
    Raises:
        ToolConfigurationError: 网关配置缺陷（path 占位符缺参等）。
        ToolExecutionError: 网络、状态码或业务码失败。
    """
    # 参数路由
    path_arguments = {
        name: _stringify_argument(arguments[name])
        for name, location in api.argument_locations.items()
        if location == "path" and name in arguments
    }
    try:
        endpoint = api.endpoint.format(**path_arguments)
    except KeyError as error:
        raise ToolConfigurationError(f"API {api.name} 的路径缺少参数占位符: {error}") from error

    query_parameters = {
        name: _stringify_argument(value)
        for name, value in arguments.items()
        if api.argument_locations.get(name) == "query"
    }
    body = {
        name: value
        for name, value in arguments.items()
        if api.argument_locations.get(name) == "body"
    }

    # 重试主循环
    max_attempts = api.retry_attempts + 1
    auth_replays_left = _MAX_AUTH_REPLAYS
    last_error: ToolExecutionError | None = None

    for attempt in range(max_attempts):
        token = await token_provider.get_valid_token(client)
        try:
            response = await client.request(
                method=api.method,
                url=f"{base_url.rstrip('/')}{endpoint}",
                params=query_parameters or None,
                json=body or None,
                headers={"Authorization": f"Bearer {token}"},
            )
        except httpx.RequestError as error:
            if attempt == max_attempts - 1:
                error_code = "upstream_timeout" if isinstance(error, httpx.TimeoutException) else "upstream_network_error"
                raise ToolExecutionError(
                    api.error_messages.get("network", "中台服务暂时不可用，请稍后重试。"),
                    error_code=error_code,
                    retryable=True,
                ) from error
            # 指数退避：给上游恢复窗口，避免紧贴重试放大故障
            await asyncio.sleep(api.retry_delay_seconds * (2**attempt))
            continue

        if response.status_code == 200:
            return _parse_success_payload(api, response)
        if response.status_code == 401 and auth_replays_left > 0:
            # token 可能已在中台侧失效：强制换新后重放，不消耗常规重试次数
            auth_replays_left -= 1
            logger.warning("gateway_http_401_replaying_after_refresh tool=%s", api.name)
            await token_provider.force_refresh(client)
            continue
        if response.status_code < 500:
            message = api.error_messages.get(str(response.status_code), "请求失败，请检查参数后重试。")
            raise ToolExecutionError(message, error_code=f"upstream_http_{response.status_code}", retryable=False)
        if attempt == max_attempts - 1:
            raise ToolExecutionError(
                api.error_messages.get("5xx", "中台服务暂时不可用，请稍后重试。"),
                error_code="upstream_server_error",
                retryable=True,
            )
        last_error = ToolExecutionError(
            api.error_messages.get("5xx", "中台服务暂时不可用，请稍后重试。"),
            error_code="upstream_server_error",
            retryable=True,
        )
        await asyncio.sleep(api.retry_delay_seconds * (2**attempt))

    raise last_error or ToolExecutionError("中台服务暂时不可用，请稍后重试。", error_code="upstream_server_error", retryable=True)

# 业务码判定
def _parse_success_payload(api: ApiConfig, response: httpx.Response) -> dict[str, Any]:
    """解析 200 响应并按 BaseResp 契约判定业务结果。

    Raises:
        ToolExecutionError: JSON 非法、结构不符或业务码非 0。
    """

    try:
        payload = response.json()
    except ValueError as error:
        raise ToolExecutionError("中台返回了无法解析的数据。", error_code="invalid_upstream_response", retryable=False) from error
    if not isinstance(payload, dict) or "code" not in payload:
        raise ToolExecutionError("中台返回了无效的数据结构。", error_code="invalid_upstream_payload", retryable=False)
    business_code = payload.get("code")
    if business_code != _SUCCESS_CODE:
        # message 来自中台，可能包含内部细节；只透传有限长度，避免泄露
        detail = str(payload.get("message", ""))[:120]
        raise ToolExecutionError(
            f"中台业务处理失败（{detail}）。" if detail else "中台业务处理失败。",
            error_code=f"upstream_business_{business_code}",
            retryable=False,
        )
    data = payload.get("data")
    return data if isinstance(data, dict) else {"result": data}
