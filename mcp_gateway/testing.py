"""测试辅助：提供一个与真实网关同契约的内存 MCP 网关。

用途：单测离线运行。`build_tools_for_agent` 通过 `_create_client` 连接网关，
测试用 `unittest.mock.patch` 替换该函数返回连接本内存网关的 Client，
即可在不启动进程、不访问真实中台的情况下覆盖完整工具链路。
"""

from __future__ import annotations

import json
from typing import Any

from fastmcp import FastMCP


def create_in_memory_gateway(
    *,
    tools: dict[str, dict[str, Any]] | None = None,
) -> FastMCP:
    """构建内存 MCP 网关，工具行为由调用方以 mock 响应声明。

    Args:
        tools: 工具名 → 定义。缺省时提供与 config/gateways.yaml 对齐的
            `nms.query_device_by_ip`；响应固定为成功样例，测试需要自定义
            响应时传入自定义 handler。
    Returns:
        可直接传给 fastmcp.Client 的 FastMCP 实例（内存传输）。
    """

    gateway = FastMCP(name="in-memory-gateway")
    for tool_name, tool_spec in (tools or _default_tools()).items():
        handler = tool_spec["handler"]

        # fastmcp 不支持 **kwargs 签名的工具；此处固定使用与真实工具一致的
        # 单参数签名（首批中台工具均为单参数），多参数工具出现时再泛化。
        @gateway.tool(
            name=tool_name,
            description=tool_spec["description"],
            meta=tool_spec.get("meta"),
        )
        async def _invoke(ip: str = "") -> str:
            """按声明的响应返回网关契约 JSON。"""

            result = handler({"ip": ip})
            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False)

    return gateway

    return gateway


def _default_tools() -> dict[str, dict[str, Any]]:
    """默认工具集：与 gateways.yaml 的 NMS 首批 5 个只读工具同名同前缀。"""

    ok = {"ok": True, "data": {"status": "online"}}
    return {
        name: {
            "description": f"{name}（内存桩）",
            "handler": lambda arguments: ok,
            "meta": {
                "requires_confirmation": name == "nms.query_device_by_ip",
                "risk_level": "low",
            },
        }
        for name in (
            "nms.query_device_by_ip",
            "nms.get_topology_graph",
            "nms.get_realtime_metric",
            "nms.get_metric_range",
            "nms.list_metric_names",
            "nms.list_alarm_events",
        )
    }
