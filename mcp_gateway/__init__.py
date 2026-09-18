"""通用 MCP 网关：把声明式配置的中台 REST API 暴露为 MCP 工具。

设计边界：本包只做协议翻译，不做业务编排；接新中台只改 config/gateways.yaml。
"""

from mcp_gateway.config_loader import GatewaySettings, load_gateway_settings

__all__ = ["GatewaySettings", "load_gateway_settings"]
