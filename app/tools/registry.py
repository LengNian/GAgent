"""把 MCP 网关声明转换为 Agent 可绑定的 LangChain Tool。

Agent 建图时从共享的 gateways.yaml 读取工具契约，实际执行时再调用 MCP 网关。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.agent_manifest import get_agent_manifest
from app.settings import Settings
from mcp_gateway.config_loader import ApiConfig, load_gateway_settings


logger = logging.getLogger(__name__)
GATEWAY_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "gateways.yaml"


def _create_client(gateway_url: str, gateway_token: str) -> Client:
    """创建连接 MCP 网关的客户端。

    独立成模块级函数：测试通过 patch 它注入内存网关（FastMCP 实例），
    避免单测依赖真实网络；生产路径保持 streamable HTTP 不变。

    Args:
        gateway_url: 网关 MCP 端点。
        gateway_token: 请求方静态 Token。
    Returns:
        fastmcp Client（未连接，调用方负责上下文管理）。
    """

    return Client(StreamableHttpTransport(gateway_url, auth=gateway_token))


def _mcp_settings(settings: Settings) -> tuple[str, str]:
    """读取网关地址与请求方 Token。

    Args:
        settings: Agent 全局配置。
    Returns:
        (网关 MCP 端点 URL, 静态 Token)。
    Raises:
        ValueError: 配置缺失。
    """

    gateway_url = settings.mcp_gateway_url
    gateway_token = settings.mcp_gateway_token.get_secret_value()
    if not gateway_url or not gateway_token:
        raise ValueError("MCP gateway URL/token is not configured")
    return gateway_url, gateway_token


def _build_args_schema(tool_name: str, parameters: dict[str, Any]) -> type[BaseModel]:
    """按网关工具的 JSON schema 构造 Pydantic 参数模型。

    逻辑规划：
    1. 遍历 properties 按 type 映射基础类型（与网关侧签名类型一致）。
    2. 依据 required 设置必填性；extra="forbid" 拒绝 schema 外参数。
    """

    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    type_mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    fields: dict[str, tuple[type[Any], Any]] = {}
    for field_name, property_config in properties.items():
        field_type = type_mapping.get(property_config.get("type"))
        if field_type is None:
            raise ValueError(f"unsupported gateway tool parameter type: {property_config.get('type')}")
        default = ... if field_name in required else None
        fields[field_name] = (
            field_type,
            Field(default=default, description=property_config.get("description", "")),
        )
    return create_model(
        f"{tool_name.title().replace('.', '').replace('_', '')}Arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _build_tool(
    gateway_url: str,
    gateway_token: str,
    tool_name: str,
    description: str,
    parameters: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> BaseTool:
    """把网关工具包成 LangChain StructuredTool。

    逻辑规划：
    1. 入参由 args_schema（来自网关 JSON schema）先行校验。
    2. 每次调用新建 MCP 会话：调用发生在 LangGraph 图执行期，
       发现期的连接早已关闭，不能复用。
    3. 网关契约 {ok, data|error_code, message} 在返回前补齐为
       ActionResult 形态（agent_id/action_name/error_type），保证
       Report 节点与 ToolNode 消费方零改动。
    4. 失败结果转成 ToolMessage 文本返回给模型，不抛异常中断图执行。
    """

    _, _, local_name = tool_name.partition(".")

    def _as_action_result(payload: dict[str, Any]) -> dict[str, Any]:
        """把网关契约补齐为 ActionResult 必需字段。"""

        payload.setdefault("agent_id", local_name)
        payload.setdefault("action_name", local_name)
        if not payload.get("ok"):
            payload.setdefault("error_type", "response")
            payload.setdefault("retryable", False)
            payload.setdefault("details", {})
        return payload

    def _parse_gateway_payload(content: object) -> dict[str, Any]:
        """从 MCP 内容块中读取网关 JSON 契约。"""

        # =========================================================================
        # [逻辑规划] MCP 返回内容兼容路径
        # =========================================================================
        # 1. 遍历所有内容块，而不是假定第一个块必为文本。
        # 2. 读取文本块的 text 字段，逐块解析网关约定的 JSON 对象。
        # 3. 没有任何合法对象时抛出 ValueError，调用方按响应契约异常处理。
        # =========================================================================
        blocks = content if isinstance(content, list) else [content]
        for block in blocks:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if not isinstance(text, str):
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("MCP tool response does not contain a JSON object")

    async def invoke_tool(**arguments: object) -> str:
        """执行网关工具调用（每次调用建立独立 MCP 会话）。"""

        try:
            async with _create_client(gateway_url, gateway_token) as client:
                result = await client.call_tool(tool_name, dict(arguments))
            payload = _parse_gateway_payload(result.content)
        except ToolError as error:
            # 网关以 is_error 结果返回契约错误（fastmcp 客户端转成 ToolError），
            # 错误消息即网关契约 JSON，解析后透传给 Report 节点转述。
            try:
                payload = json.loads(str(error))
            except (ValueError, TypeError):
                logger.exception("gateway_tool_error_unparseable tool=%s", tool_name)
                payload = {"ok": False, "error_code": "gateway_unreachable", "retryable": True, "message": "工具网关暂时不可用，请稍后重试。"}
            return json.dumps(_as_action_result(payload), ensure_ascii=False)
        except Exception:
            logger.exception("gateway_tool_call_failed tool=%s", tool_name)
            return json.dumps(
                _as_action_result(
                    {
                        "ok": False,
                        "error_code": "gateway_unreachable",
                        "error_type": "transport",
                        "retryable": True,
                        "message": "工具网关暂时不可用，请稍后重试。",
                    }
                ),
                ensure_ascii=False,
            )
        return json.dumps(_as_action_result(payload), ensure_ascii=False)

    return StructuredTool.from_function(
        coroutine=invoke_tool,
        name=tool_name,
        description=description,
        args_schema=_build_args_schema(tool_name, parameters),
        metadata=metadata or {},
    )


def _configured_tools() -> dict[str, tuple[str, ApiConfig]]:
    """读取本地网关声明，构造不依赖网络的工具契约。"""

    # =========================================================================
    # [逻辑规划] 工具契约加载路径
    # =========================================================================
    # 1. 网关与 Agent 共用 gateways.yaml，配置加载已校验 API 名称、参数和位置。
    # 2. Agent 构建只读取该声明，不在普通聊天请求中发起 MCP 网络连接。
    # 3. 实际 MCP 连通性和工具执行仍在 StructuredTool 调用期处理并安全降级。
    # =========================================================================
    settings = load_gateway_settings(GATEWAY_CONFIG_PATH)
    tools: dict[str, tuple[str, ApiConfig]] = {}
    for platform in settings.platforms:
        for api in platform.apis:
            if api.name in tools:
                raise ValueError(f"Gateway tool local name is duplicated: {api.name}")
            tools[api.name] = (platform.name, api)
    return tools


def build_tools_for_agent(settings: Settings, agent_id: str) -> list[BaseTool]:
    """构造 Agent allowlist 内的 MCP 工具。

    逻辑规划：
    1. 读取 Agent manifest 的 allowlist（仍为 `动作名`，匹配网关工具去掉
       `平台.` 前缀后的本地名）。
    2. 从已校验的 gateways.yaml 读取 MCP 工具契约，避免网关故障阻塞普通聊天。
    3. 对 allowlist 命中的工具构造 LangChain Tool；调用期再通过 MCP 执行。

    Args:
        settings: Agent 全局配置。
        agent_id: Agent 标识（决定 allowlist）。
    Returns:
        可绑定给模型的工具列表。
    Raises:
        ValueError: 网关未配置或 allowlist 与网关配置不一致。
    """

    gateway_url, gateway_token = _mcp_settings(settings)
    manifest = get_agent_manifest(agent_id)
    allowed_actions = list(manifest.allowed_actions)
    if not allowed_actions:
        return []

    tools_by_local_name = _configured_tools()

    missing = [name for name in allowed_actions if name not in tools_by_local_name]
    if missing:
        raise ValueError(f"Gateway does not expose allowed actions: {missing}")

    # 工具调用各自建立 MCP 会话：LangGraph 工具在图执行期才被调用，
    # 复用发现期客户端会因会话生命周期错配而失效。
    built_tools: list[BaseTool] = []
    for action_name in allowed_actions:
        platform_name, gateway_tool = tools_by_local_name[action_name]
        built_tools.append(
            _build_tool(
                gateway_url,
                gateway_token,
                f"{platform_name}.{gateway_tool.name}",
                gateway_tool.description or action_name,
                gateway_tool.input_schema,
                {
                    "requires_confirmation": gateway_tool.requires_confirmation,
                    "risk_level": gateway_tool.risk_level,
                },
            )
        )
    return built_tools


def build_enabled_tools(settings: Settings) -> list[BaseTool]:
    """兼容旧调用方，按默认 IoT Agent allowlist 构造工具。"""

    return build_tools_for_agent(settings, "iot_agent")
