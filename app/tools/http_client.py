"""执行已声明的只读 NMS HTTP 工具。"""

import asyncio
from typing import Any

import httpx

from app.settings import Settings

from .config import ToolConfig


class ToolExecutionError(Exception):
    """可安全展示给 Agent 和用户的工具执行错误。"""


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
        raise ToolExecutionError("工具路径缺少必要参数配置。") from error

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
            try:
                response = await client.request(
                    method=tool_config.method,
                    url=endpoint,
                    params=query_parameters,
                    json=body or None,
                )
            except httpx.RequestError as error:
                if attempt == max_attempts - 1:
                    raise ToolExecutionError(
                        tool_config.error_messages.get("network", "服务暂时不可用，请稍后重试。")
                    ) from error
            else:
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                    except ValueError as error:
                        raise ToolExecutionError("设备查询服务返回了无法解析的数据。") from error
                    if not isinstance(response_data, (dict, list)):
                        raise ToolExecutionError("设备查询服务返回了无效数据。")
                    return response_data
                if response.status_code == 400:
                    raise ToolExecutionError(tool_config.error_messages.get("400", "请求参数无效。"))
                if response.status_code == 404:
                    raise ToolExecutionError(tool_config.error_messages.get("404", "请求资源不存在。"))
                if response.status_code < 500:
                    message = tool_config.error_messages.get(str(response.status_code), "请求失败。")
                    raise ToolExecutionError(message)
                if attempt == max_attempts - 1:
                    raise ToolExecutionError(tool_config.error_messages.get("5xx", "服务暂时不可用，请稍后重试。"))

            await asyncio.sleep(tool_config.retry_delay_seconds * (2**attempt))

    raise ToolExecutionError(tool_config.error_messages.get("5xx", "服务暂时不可用，请稍后重试。"))
