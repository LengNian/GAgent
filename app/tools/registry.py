"""将 YAML 工具配置转换为 LangChain Tool。"""

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.settings import Settings

from .config import ToolConfig, get_tools_config
from .http_client import ToolExecutionError, request_tool_json


def _build_args_schema(tool_config: ToolConfig) -> type[BaseModel]:
    """根据 YAML 参数 schema 构造通用 Pydantic 参数模型。

    逻辑规划：
    1. 遍历 YAML 的 properties，按 type 映射基础 Python 类型。
    2. 依据 required 设置字段必填性，其他字段使用 None 作为默认值。
    3. 禁止未在 YAML 声明的额外参数，让模型调用严格服从配置。
    """

    properties = tool_config.parameters.get("properties", {})
    required = set(tool_config.parameters.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}
    type_mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    for name, property_config in properties.items():
        if not isinstance(property_config, dict):
            raise ValueError(f"tool parameter schema must be an object: {name}")
        field_type = type_mapping.get(property_config.get("type"))
        if field_type is None:
            raise ValueError(f"unsupported tool parameter type: {property_config.get('type')}")
        default = ... if name in required else None
        fields[name] = (field_type, Field(default=default, description=property_config.get("description", "")))
    return create_model(
        f"{tool_config.name.title().replace('_', '')}Arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


async def _execute_configured_tool(
    tool_config: ToolConfig,
    arguments: dict[str, object],
    settings: Settings,
) -> str:
    """执行 YAML 配置的 HTTP Tool 并返回原始 JSON。

    逻辑规划：
    1. 根据 YAML 的 argument_locations 将参数放入 query、path 或 body。
    2. 调用统一 HTTP 客户端处理请求、超时、重试和状态码。
    3. 保留后端 JSON 结构，不为某个业务 API 写死响应字段或数量限制。
    4. 遇到可预期工具错误时返回安全错误文本，供模型生成最终答复。
    """

    try:
        response_data = await request_tool_json(tool_config, arguments, settings)
    except ToolExecutionError as error:
        return json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False)
    return json.dumps(response_data, ensure_ascii=False)


def _build_tool(tool_config: ToolConfig, settings: Settings) -> BaseTool:
    """将一项已启用 YAML 配置构造成可绑定给模型的 Tool。

    逻辑规划：
    1. 根据 YAML 构造参数 schema，确保模型调用参数先经过 Pydantic 校验。
    2. 将当前工具配置和 Settings 捕获在协程中，避免模型传入地址、方法或密钥。
    3. 用 YAML 中的名称与描述创建 StructuredTool，供模型选择调用。
    """

    async def invoke_tool(**arguments: object) -> str:
        """执行当前 YAML 工具的运行时调用。

        逻辑规划：
        1. 接收 StructuredTool 已按 YAML schema 校验的参数。
        2. 交给通用 HTTP 执行函数按 YAML 的请求位置发送。
        3. 将成功 JSON 或安全错误转换为模型可读取的文本。
        """

        return await _execute_configured_tool(tool_config, arguments, settings)

    return StructuredTool.from_function(
        coroutine=invoke_tool,
        name=tool_config.name,
        description=tool_config.description,
        args_schema=_build_args_schema(tool_config),
    )


def build_enabled_tools(settings: Settings) -> list[BaseTool]:
    """构造当前 YAML 中所有启用的 LangChain Tool。

    逻辑规划：
    1. 读取并校验 tools.yaml。
    2. 忽略 disabled 工具，确保未启用接口不会绑定给模型。
    3. 将每项启用配置转换为 StructuredTool，并返回给 Agent 工厂。
    """

    return [_build_tool(tool, settings) for tool in get_tools_config().tools if tool.enabled]
