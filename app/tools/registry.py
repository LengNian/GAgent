"""将 YAML 工具配置转换为 LangChain Tool。"""

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, create_model

from app.action_gateway import ActionGateway
from app.agent_manifest import get_agent_manifest
from app.ontology import ActionConfig, get_action_registry
from app.settings import Settings

from .config import get_tools_config


def _build_args_schema(name: str, parameter_schema: dict[str, Any]) -> type[BaseModel]:
    """根据 Action 参数 schema 构造通用 Pydantic 参数模型。

    逻辑规划：
    1. 遍历 YAML 的 properties，按 type 映射基础 Python 类型。
    2. 依据 required 设置字段必填性，其他字段使用 None 作为默认值。
    3. 禁止未在 YAML 声明的额外参数，让模型调用严格服从配置。
    """

    properties = parameter_schema.get("properties", {})
    required = set(parameter_schema.get("required", []))
    fields: dict[str, tuple[type[Any], Any]] = {}
    type_mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }
    for name, property_config in properties.items():
        if not isinstance(property_config, dict):
            raise ValueError(f"Action parameter schema must be an object: {name}")
        field_type = type_mapping.get(property_config.get("type"))
        if field_type is None:
            raise ValueError(f"unsupported Action parameter type: {property_config.get('type')}")
        default = ... if name in required else None
        fields[name] = (field_type, Field(default=default, description=property_config.get("description", "")))

    # 运行时动态创建一个Pydantic模型
    return create_model(
        f"{name.title().replace('_', '')}Arguments",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _build_tool(action: ActionConfig, gateway: ActionGateway) -> BaseTool:
    """将一个已授权 Action 和其执行器构造成可绑定给模型的 Tool。
       把一个"已授权的 Action + 它的网关"包装成 StructuredTool

    逻辑规划：
    1. 使用 Ontology Action 的 schema，确保模型只能生成业务契约声明的参数。
    2. 将绑定 Agent 身份的 Gateway 捕获在协程中，避免模型伪造执行身份。
    3. 使用 Action 名称和描述创建 StructuredTool，底层 executor 只由 Gateway 调用。
    """

    async def invoke_tool(**arguments: object) -> str:
        """执行当前 YAML 工具的运行时调用。

        逻辑规划：
        1. 接收 StructuredTool 已按 YAML schema 校验的参数。
        2. 交给通用 HTTP 执行函数按 YAML 的请求位置发送。
        3. 将成功 JSON 或安全错误转换为模型可读取的文本。
        """

        return await gateway.execute(action.name, arguments)

    return StructuredTool.from_function(
        coroutine=invoke_tool,
        name=action.name,
        description=action.description,
        args_schema=_build_args_schema(action.name, action.input_schema),
    )


def build_tools_for_agent(settings: Settings, agent_id: str) -> list[BaseTool]:
    """只构造指定 Agent manifest allowlist 中的工具。

    逻辑规划：
    1. 读取已校验的 Agent manifest，获得明确的 Action allowlist。
    2. 通过 Action Registry 将 Action 映射到声明的底层 executor。
    3. 只为 allowlist 中且 executor 已启用的 Action 构造 StructuredTool。
    4. 若出现配置不一致立即失败，避免以全量工具或空工具静默降级。
    """

    # 这里得到的就是agen.yaml
    manifest = get_agent_manifest(agent_id)

    # 这里得到的就是actions.yaml
    action_registry = get_action_registry()
    tools_by_name = {
        tool.name: tool for tool in get_tools_config().tools if tool.enabled
    }

    gateway = ActionGateway(agent_id=agent_id, settings=settings)

    built_tools: list[BaseTool] = []

    for action_name in manifest.allowed_actions:

        action = action_registry.get(action_name)
        tool_config = tools_by_name.get(action.executor)

        if tool_config is None:
            raise ValueError(
                f"Action {action_name} executor is not enabled: {action.executor}"
            )
        built_tools.append(_build_tool(action, gateway))

    return built_tools


def build_enabled_tools(settings: Settings) -> list[BaseTool]:
    """兼容旧调用方，按默认 IoT Agent allowlist 构造工具。"""

    return build_tools_for_agent(settings, "iot_agent")
