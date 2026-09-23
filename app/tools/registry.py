"""把 MCP Gateway 发现的工具转换为 Agent 可绑定的 LangChain Tool。"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.agent_manifest import get_agent_manifest
from app.settings import Settings


logger = logging.getLogger(__name__)


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
    1. 遍历 properties，兼容 MCP 对可选参数生成的 `anyOf[type, null]` Schema。
    2. 按基础 type 映射 Pydantic 类型，未知或复杂联合类型明确拒绝。
    3. 依据 required 设置必填性；extra="forbid" 拒绝 schema 外参数。
    """

    properties = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    type_mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }

    def resolve_property_type(property_config: dict[str, Any]) -> type[Any] | None:
        """解析 MCP 工具参数的基础 JSON Schema 类型。

        逻辑规划：
        1. 优先读取直接声明的 `type`。
        2. 缺失时识别 FastMCP 为 `T | None` 生成的 `anyOf[T, null]`。
        3. 仅接受唯一的非 null 基础类型，避免将复杂 schema 错误降级为字符串。
        """

        schema_type = property_config.get("type")
        if isinstance(schema_type, str):
            return type_mapping.get(schema_type)
        variants = property_config.get("anyOf")
        if not isinstance(variants, list):
            return None
        resolved_types = {
            variant.get("type")
            for variant in variants
            if isinstance(variant, dict) and variant.get("type") != "null"
        }
        if len(resolved_types) != 1:
            return None
        schema_type = resolved_types.pop()
        return type_mapping.get(schema_type) if isinstance(schema_type, str) else None

    fields: dict[str, tuple[type[Any], Any]] = {}
    for field_name, property_config in properties.items():
        if not isinstance(property_config, dict):
            raise ValueError(f"invalid gateway tool parameter schema: {field_name}")
        field_type = resolve_property_type(property_config)
        if field_type is None:
            raise ValueError(f"unsupported gateway tool parameter schema: {property_config}")
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

    # 模型可见的工具名必须是裸本地名：MCP 全名带平台前缀（如 nms.query_device_by_ip），
    # 而部分 OpenAI 兼容供应商严格按 ^[a-zA-Z0-9_-]+$ 校验 tools[].function.name，
    # 点号会导致 400。裸名同时与 agents.yaml 的 allowlist、SKILL.md 引用保持一致；
    # call_tool 仍使用带前缀全名作为 MCP 侧唯一标识。
    _, separator, local_name = tool_name.partition(".")
    if not separator:
        local_name = tool_name

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
        name=local_name,
        description=description,
        args_schema=_build_args_schema(tool_name, parameters),
        metadata=metadata or {},
    )


async def build_tools_for_agent(settings: Settings, agent_id: str) -> list[BaseTool]:
    """构造 Agent allowlist 内的 MCP 工具。

    逻辑规划：
    1. 读取 Agent manifest 的 allowlist（动作名匹配 MCP 工具去掉平台前缀后的名称）。
    2. 通过 MCP `tools/list` 获取名称、描述、输入 schema 与元数据；Agent 不读取
       Gateway 的 REST API 映射配置。
    3. 对 allowlist 命中的工具构造 LangChain Tool；调用期再通过 MCP 执行。

    Args:
        settings: Agent 全局配置。
        agent_id: Agent 标识（决定 allowlist）。
    Returns:
        可绑定给模型的工具列表。
    Raises:
        ValueError: 网关不可达或 allowlist 与 MCP 工具清单不一致。
    """

    manifest = get_agent_manifest(agent_id)
    allowed_actions = list(manifest.allowed_actions)
    if not allowed_actions:
        return []
    gateway_url, gateway_token = _mcp_settings(settings)

    # =========================================================================
    # [逻辑规划] MCP 工具发现与授权
    # =========================================================================
    # 1. 使用独立短生命周期会话调用 tools/list，防止发现会话与执行会话混用。
    # 2. 以 MCP 工具的全名作为唯一标识，本地名仅用于 agents.yaml 授权匹配。
    # 3. 缺失白名单工具时拒绝建图，避免模型获得不可执行的伪能力。
    # =========================================================================
    try:
        async with _create_client(gateway_url, gateway_token) as client:
            discovered_tools = await client.list_tools()
    except Exception as error:
        raise ValueError("Unable to discover MCP tools from gateway") from error

    tools_by_local_name: dict[str, Any] = {}
    for gateway_tool in discovered_tools:
        tool_name = str(gateway_tool.name)
        _, separator, local_name = tool_name.partition(".")
        if not separator or not local_name:
            continue
        if local_name in tools_by_local_name:
            raise ValueError(f"MCP tool local name is duplicated: {local_name}")
        tools_by_local_name[local_name] = gateway_tool

    missing = [name for name in allowed_actions if name not in tools_by_local_name]
    if missing:
        raise ValueError(f"Gateway does not expose allowed actions: {missing}")

    built_tools: list[BaseTool] = []
    for action_name in allowed_actions:
        gateway_tool = tools_by_local_name[action_name]
        metadata = getattr(gateway_tool, "meta", None) or {}
        built_tools.append(
            _build_tool(
                gateway_url,
                gateway_token,
                str(gateway_tool.name),
                str(gateway_tool.description or action_name),
                dict(gateway_tool.input_schema or {}),
                {
                    "requires_confirmation": bool(metadata.get("requires_confirmation", False)),
                    "risk_level": str(metadata.get("risk_level", "low")),
                },
            )
        )
    return built_tools


async def build_enabled_tools(settings: Settings) -> list[BaseTool]:
    """兼容旧调用方，按默认 IoT Agent allowlist 构造工具。"""

    return await build_tools_for_agent(settings, "iot_agent")
