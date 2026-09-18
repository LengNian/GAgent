"""MCP 网关服务组装：把配置中的中台 API 注册为 MCP 工具并挂上请求方鉴权。

启动方式：python -m mcp_gateway.server（streamable HTTP，默认 127.0.0.1:8001）。
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth.auth import AccessToken, TokenVerifier
from fastmcp.tools import FunctionTool
from mcp.types import CallToolResult, TextContent

from mcp_gateway.auth import LoginTokenProvider
from mcp_gateway.config_loader import (
    ApiConfig,
    GatewaySettings,
    PlatformConfig,
    load_gateway_settings,
    resolve_env_reference,
)
from mcp_gateway.errors import GatewayError, ToolAuthError
from mcp_gateway.executor import execute_api_call

logger = logging.getLogger(__name__)

# Python 类型注解 → 动态函数签名用；FastMCP 按函数签名（而非 JSON schema）校验入参
_ANNOTATION_BY_TYPE: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


class StaticTokenVerifier(TokenVerifier):
    """Agent → 网关的静态 Token 校验（Token 从环境变量读取，不落配置文件）。"""

    def __init__(self) -> None:
        super().__init__()
        import os

        self._expected_token = os.environ.get("MCP_GATEWAY_TOKEN", "")
        if not self._expected_token:
            raise RuntimeError("MCP_GATEWAY_TOKEN 环境变量未设置，网关拒绝启动")

    async def verify_token(self, token: str) -> AccessToken | None:
        """校验请求方静态 Token。

        Args:
            token: 请求 Bearer 头中的 Token。
        Returns:
            合法时返回 AccessToken，否则 None（框架回 401）。
        """

        if token == self._expected_token:
            return AccessToken(token=token, client_id="agent", scopes=[], expires_at=None)
        return None


def _build_tool_function(
    *,
    platform: PlatformConfig,
    api: ApiConfig,
    base_url: str,
    token_provider: LoginTokenProvider,
    client: httpx.AsyncClient,
):
    """为一条 API 构造具名签名的异步工具函数。

    逻辑规划：
    1. [签名构造] 依据 input_schema 动态生成 `def f(ip: str, metric: int)` 形式的函数——
       FastMCP 按函数签名而非 JSON schema 校验入参，签名字段缺失即失去校验。
    2. [执行] 委托 executor.execute_api_call；将网关错误转为 ToolError 让 MCP 层
       以错误结果返回客户端（is_error=True），而不是中断会话。
    3. [结果] 返回 JSON 字符串（结构化数据在 agent 侧按 ActionResult 消费）。

    Args:
        platform: 所属中台配置。
        api: API 映射配置。
        base_url: 已解析的中台地址。
        token_provider: 中台 token 管理器。
        client: 复用的 httpx 客户端。
    Returns:
        可注册到 FastMCP 的异步函数。
    """

    properties = api.input_schema.get("properties", {})
    required_parameters = set(api.input_schema.get("required", []))
    required_parameter_lines = []
    optional_parameter_lines = []
    for parameter_name, property_config in properties.items():
        annotation = _ANNOTATION_BY_TYPE.get(property_config.get("type", "string"), str)
        if parameter_name in required_parameters:
            required_parameter_lines.append(f"{parameter_name}: {annotation.__name__}")
        else:
            # 必填参数必须排在可选参数前，保证动态生成的 Python 签名合法；
            # None 表示调用方未提供该字段，执行前会从 HTTP 参数中剔除。
            optional_parameter_lines.append(f"{parameter_name}: {annotation.__name__} | None = None")
    parameter_lines = [*required_parameter_lines, *optional_parameter_lines]
    signature = ", ".join(parameter_lines)

    namespace: dict[str, Any] = {
        "_execute": execute_api_call,
        "_api": api,
        "_platform": platform,
        "_base_url": base_url,
        "_provider": token_provider,
        "_client": client,
        "_httpx": httpx,
        "_GatewayError": GatewayError,
        "_ToolError": ToolError,
        "_json": __import__("json"),
        **_ANNOTATION_BY_TYPE,
    }
    function_source = (
        "async def _tool_fn("
        + signature
        + ") -> str:\n"
        + "    try:\n"
        + "        result = await _execute(platform=_platform, api=_api, base_url=_base_url, "
        "token_provider=_provider, client=_client, "
        "arguments={name: value for name, value in locals().items() if value is not None})\n"
        + "    except _GatewayError as error:\n"
        + "        raise _ToolError(_json.dumps("
        "{'ok': False, 'error_code': error.error_code, 'retryable': error.retryable, "
        "'message': error.message}, ensure_ascii=False)) from error\n"
        + "    return _json.dumps({'ok': True, 'data': result}, ensure_ascii=False)\n"
    )
    exec(function_source, namespace)  # noqa: S102 - 签名由服务端配置生成，非用户输入
    return namespace["_tool_fn"]


def create_gateway_app(config_path: str = "config/gateways.yaml") -> FastMCP:
    """创建网关 FastMCP 应用。

    逻辑规划：
    1. [配置] 加载并校验 gateways.yaml，坏配置直接抛错阻止启动。
    2. [鉴权] 挂静态 Token 校验器（Agent → 网关）。
    3. [注册] 为每个中台创建 token 管理器与 httpx 客户端，把每条 API
       注册为 `平台名.工具名` 的 MCP 工具，requires_confirmation 写入 meta。
    Args:
        config_path: gateways.yaml 路径。
    Returns:
        可用 uvicorn/http_app 运行的 FastMCP 实例。
    Raises:
        RuntimeError: 网关 Token 未配置。
        ValidationError: 配置不合法。
    """

    settings: GatewaySettings = load_gateway_settings(config_path)
    clients: list[httpx.AsyncClient] = []

    @asynccontextmanager
    async def gateway_lifespan(_server: FastMCP):
        """在网关停止时关闭复用的中台 HTTP 客户端。"""

        # =========================================================================
        # [逻辑规划] 生命周期资源回收
        # =========================================================================
        # 1. 启动阶段交出已注册工具使用的 HTTP 客户端。
        # 2. 停止阶段逐个关闭客户端，释放 keep-alive 连接与连接池资源。
        # 3. 单个关闭失败不阻塞其余客户端回收。
        # =========================================================================
        try:
            yield
        finally:
            results = await asyncio.gather(
                *(client.aclose() for client in clients),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("gateway_http_client_close_failed", exc_info=result)

    mcp = FastMCP(
        name="platform-gateway",
        auth=StaticTokenVerifier(),
        lifespan=gateway_lifespan,
    )

    for platform in settings.platforms:
        base_url = settings.resolve_base_url(platform)
        token_provider = LoginTokenProvider(base_url=base_url, auth=platform.auth)
        client = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        clients.append(client)
        for api in platform.apis:
            tool_function = _build_tool_function(
                platform=platform,
                api=api,
                base_url=base_url,
                token_provider=token_provider,
                client=client,
            )
            tool = FunctionTool.from_function(
                fn=tool_function,
                name=f"{platform.name}.{api.name}",
                description=api.description,
                meta={
                    "requires_confirmation": api.requires_confirmation,
                    "risk_level": api.risk_level,
                },
            )
            mcp.add_tool(tool)
            logger.info("gateway_tool_registered name=%s", f"{platform.name}.{api.name}")
    return mcp


def main() -> None:
    """网关进程入口：启动 streamable HTTP 服务。"""

    import os

    # 与 Agent 应用共用同一份 .env；网关是独立进程，必须自行加载
    from dotenv import load_dotenv

    load_dotenv(os.environ.get("GATEWAY_ENV_PATH", "config/.env"))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = create_gateway_app(
        config_path=os.environ.get("GATEWAY_CONFIG_PATH", "config/gateways.yaml")
    )
    asgi_app = app.http_app()
    import uvicorn

    uvicorn.run(
        asgi_app,
        host=os.environ.get("GATEWAY_HOST", "127.0.0.1"),
        port=int(os.environ.get("GATEWAY_PORT", "8001")),
        lifespan="on",
    )


if __name__ == "__main__":
    main()
