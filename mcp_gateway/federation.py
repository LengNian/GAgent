"""把外部 MCP Server 的工具代理注册进网关，统一为网关的 {ok, data} 结果契约。

设计要点：
1. 网关作为上游 MCP Server 的客户端，启动时（lifespan 内）发现其工具清单，
   以 `服务名.工具名` 的形式重新暴露为本地工具，Agent 侧通过既有的 tools/list
   发现机制无感知地消费——与 REST 中台工具走完全相同的下游链路。
2. 上游工具的入参 schema 原样透传给 FastMCP，因此入参校验由 FastMCP 按 schema 完成
   （含 array 等复合类型，不受 REST 侧动态签名只支持标量的限制）。
3. 上游返回的是各自格式的文本/结构化内容，与网关约定的 {ok, data} JSON 契约不一致；
   故在代理层统一包装，保证 registry.py 的 _parse_gateway_payload 能正常解析。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import SSETransport, StdioTransport
from fastmcp.server.providers.proxy import ProxyTool
from fastmcp.tools import ToolResult
from mcp.types import TextContent

from mcp_gateway.config_loader import McpServerConfig, resolve_env_reference

logger = logging.getLogger(__name__)


class GatewayProxyTool(ProxyTool):
    """代理上游 MCP 工具，并把上游返回统一包装为网关 {ok, data} 契约。

    复用 FastMCP ProxyTool 的转发能力：`_backend_name` 保留上游原始工具名，
    对外 `name` 带服务前缀，因此调用会以 `服务名.工具名` 暴露、却仍转发到上游原名。
    """

    async def run(
        self,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> ToolResult:
        """执行上游调用并包装结果。

        Args:
            arguments: 已通过 FastMCP schema 校验的工具入参。
            context: FastMCP 请求上下文，透传给父类转发逻辑。
        Returns:
            ToolResult：content 为网关契约 JSON 文本（成功 {ok:true,data:...}，
            失败 {ok:false,error_code,retryable,message}）。
        说明：
            这里刻意不抛异常，而是把失败转成契约结果，交由 Agent 侧 ToolNode 以
            ToolMessage 文本回传给模型，与 REST 工具的错误处理行为保持一致。
        """

        backend_name = self._backend_name or self.name
        try:
            result = await super().run(arguments, context)
        except Exception as error:  # noqa: BLE001 - 上游不可达/子进程异常需兜底为契约错误
            logger.warning("federation_call_failed tool=%s error=%s", backend_name, error)
            return _text_result(
                {
                    "ok": False,
                    "error_code": "upstream_unreachable",
                    "retryable": True,
                    "message": "外部工具服务暂时不可用，请稍后重试。",
                }
            )
        text = _extract_text(result.content)
        if getattr(result, "is_error", False):
            return _text_result(
                {
                    "ok": False,
                    "error_code": "upstream_tool_error",
                    "retryable": False,
                    "message": text or "外部工具返回错误。",
                }
            )
        return _text_result({"ok": True, "data": {"text": text}})


def _extract_text(content: Any) -> str:
    """从 MCP 内容块中拼接所有文本块。"""

    blocks = content if isinstance(content, list) else [content]
    parts = [getattr(block, "text", "") for block in blocks if getattr(block, "text", None)]
    return "\n".join(part for part in parts if part)


def _text_result(payload: dict[str, Any]) -> ToolResult:
    """把契约字典序列化为单个文本块结果。"""

    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, ensure_ascii=False))]
    )


def _make_transport(config: McpServerConfig) -> StdioTransport | SSETransport:
    """按配置构造上游 MCP 客户端 transport。

    Args:
        config: 外部 MCP Server 配置。
    Returns:
        StdioTransport（fork 子进程）或 SSETransport（连接已部署端点）。
    Raises:
        EnvReferenceError: env 引用变量缺失（由 resolve_env_reference 抛出）。
    """

    if config.transport == "stdio":
        resolved_env = {
            key: resolve_env_reference(value, context=f"{config.name}.env.{key}")
            for key, value in config.env.items()
        }
        return StdioTransport(
            command=config.command,  # type: ignore[arg-type] - 校验器保证 stdio 必有 command
            args=list(config.args),
            env=resolved_env or None,
        )
    url = resolve_env_reference(config.url, context=f"{config.name}.url")  # type: ignore[arg-type]
    return SSETransport(url=url)


async def build_federation_tools(config: McpServerConfig) -> list[GatewayProxyTool]:
    """发现上游工具并构造带服务前缀的代理工具列表。

    逻辑规划：
    1. [发现] 用一次性会话连接上游，拉取工具清单（名称、描述、入参 schema）。
    2. [构造] 为每个上游工具建 GatewayProxyTool：对外名 `服务名.工具名`，
       内部 _backend_name 保留上游原名，调用时 factory 现取新 transport 转发。
    3. [隔离] 每次调用使用独立 transport，避免跨事件循环复用子进程带来的生命周期耦合。

    Args:
        config: 外部 MCP Server 配置。
    Returns:
        可 add_tool 到网关的代理工具列表。
    Raises:
        Exception: 上游连接/发现失败（调用方 lifespan 负责捕获并跳过，不阻断网关启动）。
    """

    async with Client(_make_transport(config)) as client:
        upstream_tools = await client.list_tools()

    def client_factory() -> Client:
        return Client(_make_transport(config))

    proxy_tools: list[GatewayProxyTool] = []
    for tool in upstream_tools:
        proxy = GatewayProxyTool(
            client_factory=client_factory,
            name=f"{config.name}.{tool.name}",
            description=tool.description or tool.name,
            parameters=dict(tool.input_schema or {}),
        )
        # 保留上游原始工具名，转发调用时用它命中上游；对外名已带服务前缀。
        proxy._backend_name = tool.name
        proxy_tools.append(proxy)
    logger.info(
        "federation_tools_registered server=%s count=%s", config.name, len(proxy_tools)
    )
    return proxy_tools
