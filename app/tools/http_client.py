"""执行已声明的只读 NMS HTTP 工具。"""

import asyncio
import logging
from typing import Any

import httpx

from app.action_errors import (
    ActionInternalError,
    ActionResponseError,
    ActionTransportError,
)
from app.observability import log_event
from app.settings import Settings

from .config import ToolConfig


logger = logging.getLogger(__name__)


async def request_tool_json(
    tool_config: ToolConfig,
    arguments: dict[str, object],
    settings: Settings,
) -> dict[str, Any] | list[Any]:
    """请求 YAML 声明的 HTTP 接口并返回 JSON 响应。

    逻辑规划：
    1. 根据 YAML 声明的位置，将参数拆分到 path、query 或 JSON body。
    2. 对网络错误和 5xx 响应按配置进行有限指数退避重试；400、404 不重试。
    3. 按 YAML 中配置的状态码消息转换错误，隐藏底层 HTTP 细节。
    4. 仅在 200 响应为合法 JSON 时返回，其他情况统一转换为安全错误。
    """

    path_arguments = {
        name: str(arguments[name])
        for name, location in tool_config.argument_locations.items()
        if location == "path" and name in arguments
    }

    try:
        endpoint = tool_config.endpoint.format(**path_arguments)
    except KeyError as error:
        raise ActionInternalError(
            "工具路径缺少必要参数配置。",
            error_code="tool_configuration_error",
        ) from error

    query_parameters = {
        name: value
        for name, value in arguments.items()
        if tool_config.argument_locations.get(name) == "query"
    }

    body = {
        name: value
        for name, value in arguments.items()
        if tool_config.argument_locations.get(name) == "body"
    }

    timeout = httpx.Timeout(tool_config.timeout_seconds)
    max_attempts = tool_config.retry_attempts + 1

    async with httpx.AsyncClient(base_url=settings.nms_api_base_url, timeout=timeout) as client:
        for attempt in range(max_attempts):
            log_event(
                logger,
                logging.INFO,
                "http_attempt_started",
                method=tool_config.method,
                endpoint=endpoint,
                attempt=attempt + 1,
                max_attempts=max_attempts,
            )
            try:
                response = await client.request(
                    method=tool_config.method,
                    url=endpoint,
                    params=query_parameters,
                    json=body or None,
                )
            except httpx.RequestError as error:
                if attempt == max_attempts - 1:
                    error_code = (
                        "upstream_timeout"
                        if isinstance(error, httpx.TimeoutException)
                        else "upstream_network_error"
                    )
                    raise ActionTransportError(
                        tool_config.error_messages.get("network", "服务暂时不可用，请稍后重试。"),
                        retryable=True,
                        error_code=error_code,
                    ) from error
                log_event(
                    logger,
                    logging.WARNING,
                    "http_attempt_retry_scheduled",
                    method=tool_config.method,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    delay_seconds=tool_config.retry_delay_seconds * (2**attempt),
                    error_type="transport",
                )
            else:
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                    except ValueError as error:
                        raise ActionResponseError(
                            "设备查询服务返回了无法解析的数据。",
                            error_code="invalid_upstream_response",
                        ) from error
                    if not isinstance(response_data, (dict, list)):
                        raise ActionResponseError(
                            "设备查询服务返回了无效数据。",
                            error_code="invalid_upstream_payload",
                        )
                    return response_data
                if response.status_code == 400:
                    raise ActionResponseError(
                        tool_config.error_messages.get("400", "请求参数无效。"),
                        error_code="upstream_bad_request",
                    )
                if response.status_code == 404:
                    raise ActionResponseError(
                        tool_config.error_messages.get("404", "请求资源不存在。"),
                        error_code="upstream_not_found",
                    )
                if response.status_code < 500:
                    message = tool_config.error_messages.get(str(response.status_code), "请求失败。")
                    raise ActionResponseError(
                        message,
                        error_code=f"upstream_http_{response.status_code}",
                    )
                if attempt == max_attempts - 1:
                    raise ActionTransportError(
                        tool_config.error_messages.get("5xx", "服务暂时不可用，请稍后重试。"),
                        retryable=True,
                        error_code="upstream_server_error",
                    )
                log_event(
                    logger,
                    logging.WARNING,
                    "http_attempt_retry_scheduled",
                    method=tool_config.method,
                    endpoint=endpoint,
                    attempt=attempt + 1,
                    next_attempt=attempt + 2,
                    delay_seconds=tool_config.retry_delay_seconds * (2**attempt),
                    error_type="transport",
                    error_code="upstream_server_error",
                    status_code=response.status_code,
                )

            await asyncio.sleep(tool_config.retry_delay_seconds * (2**attempt))
