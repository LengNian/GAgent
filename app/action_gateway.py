"""Action 的服务端授权、参数校验和执行边界。
   注意：现在这里的precondition_validators写死了，只校验ipv4，后续还得补充内容
"""

import ipaddress
import json
from typing import Any

from app.agent_manifest import get_agent_manifest
from app.ontology import ActionConfig, get_action_registry
from app.settings import Settings
from app.tools.config import get_tools_config
from app.tools.http_client import ToolExecutionError, request_tool_json


class ActionGatewayError(Exception):
    """Action 未通过 Gateway 校验的安全错误。"""


class ActionGateway:
    """校验 Agent Action 权限并调用已注册的底层执行器。"""

    def __init__(self, *, agent_id: str, settings: Settings) -> None:
        """创建绑定到一个 Agent 身份的 Gateway。

        逻辑规划：
        1. 读取并缓存 Agent manifest，固定本次 Gateway 的 Agent 身份。
        2. 读取 Action Registry 和启用的执行器，建立只读执行映射。
        3. 每次 execute 仍重新检查 Action allowlist，避免调用方绕过边界。
        """

        self.agent_id = agent_id
        self.settings = settings
        self.manifest = get_agent_manifest(agent_id)
        self.action_registry = get_action_registry()
        self.executors = {
            tool.name: tool for tool in get_tools_config().tools if tool.enabled
        }

    async def execute(self, action_name: str, arguments: dict[str, object]) -> str:
        """校验并执行一个 Action，返回供 Agent 使用的安全 JSON 文本。

        逻辑规划：
        1. 检查 Action 是否注册且属于当前 Agent allowlist。
        2. 校验参数类型、必填字段和额外字段，拒绝任意输入。
        3. 执行 Ontology 声明的确定性前置条件；当前支持 IPv4 校验器。
        4. 根据 Action executor 找到底层 HTTP 配置并调用统一客户端。
        5. 将预期工具失败转换为 Agent 可理解的安全错误。
        """

        try:

            try:
                action = self.action_registry.get(action_name)
            except KeyError as error:
                raise ActionGatewayError(str(error)) from error

            self._authorize(action)

            validated_arguments = self._validate_arguments(action, arguments)
            self._validate_preconditions(action, validated_arguments)
            executor = self.executors.get(action.executor)
            if executor is None:
                raise ActionGatewayError(f"Action executor is not enabled: {action.executor}")
            response_data = await request_tool_json(
                executor,
                validated_arguments,
                self.settings,
            )
        except ActionGatewayError as error:
            return json.dumps(
                {
                    "ok": False,
                    "agent_id": self.agent_id,
                    "action_name": action_name,
                    "retryable": False,
                    "error_type": "gateway_validation",
                    "message": str(error),
                    "error": str(error),
                },
                ensure_ascii=False,
            )
        except ToolExecutionError as error:
            return json.dumps(
                {
                    "ok": False,
                    "agent_id": self.agent_id,
                    "action_name": action_name,
                    "retryable": error.retryable,
                    "error_type": error.error_type,
                    "message": str(error),
                    "error": str(error),
                },
                ensure_ascii=False,
            )
        return json.dumps(response_data, ensure_ascii=False)

    def _authorize(self, action: ActionConfig) -> None:
        """确认当前 Agent 被允许执行指定 Action。"""

        if action.name not in self.manifest.allowed_actions:
            raise ActionGatewayError(
                f"Agent {self.agent_id} is not authorized for Action {action.name}"
            )

    @staticmethod
    def _validate_arguments(
        action: ActionConfig,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        """按 Action schema 校验参数并返回未修改的安全副本。"""

        if not isinstance(arguments, dict):
            raise ActionGatewayError("Action arguments must be an object")

        schema = action.input_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        missing = required - set(arguments)
        unknown = set(arguments) - set(properties)

        if missing:
            raise ActionGatewayError(f"Action arguments missing required fields: {sorted(missing)}")
        if unknown:
            raise ActionGatewayError(f"Action arguments contain unknown fields: {sorted(unknown)}")

        type_mapping: dict[str, type[Any] | tuple[type[Any], ...]] = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
        }

        for field_name, value in arguments.items():
            expected_type = type_mapping.get(properties[field_name].get("type"))
            if expected_type is None:
                raise ActionGatewayError(f"Unsupported Action parameter type: {field_name}")
            if isinstance(value, bool) and expected_type != bool:
                raise ActionGatewayError(f"Action parameter has invalid type: {field_name}")
            if not isinstance(value, expected_type):
                raise ActionGatewayError(f"Action parameter has invalid type: {field_name}")
        return dict(arguments)

    @staticmethod
    def _validate_preconditions(
        action: ActionConfig,
        arguments: dict[str, object],
    ) -> None:
        """执行 Action 声明的确定性本地前置条件。"""

        validators = {"ipv4": ActionGateway._validate_ipv4}
        for precondition in action.preconditions:
            validator = validators.get(precondition.name)
            if validator is None:
                raise ActionGatewayError(
                    f"Unsupported Action precondition validator: {precondition.name}"
                )
            if precondition.field not in arguments:
                raise ActionGatewayError(
                    f"Action precondition field is missing: {precondition.field}"
                )
            validator(arguments[precondition.field], precondition.field)

    @staticmethod
    def _validate_ipv4(value: object, field_name: str) -> None:
        """验证字段是合法 IPv4 地址。"""

        if not isinstance(value, str):
            raise ActionGatewayError(f"Action parameter must be IPv4 text: {field_name}")
        try:
            address = ipaddress.ip_address(value)
        except ValueError as error:
            raise ActionGatewayError(
                f"Action parameter is not a valid IPv4 address: {field_name}"
            ) from error
        if address.version != 4:
            raise ActionGatewayError(f"Action parameter must be IPv4: {field_name}")
