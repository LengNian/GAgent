"""读取并校验 YAML 工具配置。"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


TOOLS_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "tools.yaml"


class ToolConfig(BaseModel):
    """一个由 YAML 驱动的 HTTP Tool 配置。"""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    enabled: bool
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"]
    endpoint: str = Field(pattern=r"^/")
    timeout_seconds: float = Field(gt=0, le=60)
    retry_attempts: int = Field(ge=0, le=3)
    retry_delay_seconds: float = Field(gt=0, le=10)
    parameters: dict[str, Any] = Field(default_factory=dict)
    argument_locations: dict[str, Literal["query", "path", "body"]] = Field(default_factory=dict)
    error_messages: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameter_schema(self) -> "ToolConfig":
        """校验 YAML 参数 schema 与请求位置配置的一致性。

        逻辑规划：
        1. 允许 parameters 为空，支持无参数接口。
        2. 有参数时确认 type 为 object，并要求 properties 和 required 使用标准结构。
        3. 确认每个请求位置都对应一个已声明参数，避免运行时静默丢参。
        4. 不限制参数名称或具体业务类型，参数变化只修改 YAML。
        """

        if not self.parameters:
            if self.argument_locations:
                raise ValueError("argument_locations cannot be set without parameters")
            return self

        if self.parameters.get("type") != "object":
            raise ValueError("parameters.type must be object")
        properties = self.parameters.get("properties")
        required = self.parameters.get("required", [])
        if not isinstance(properties, dict):
            raise ValueError("parameters.properties must be an object")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError("parameters.required must be a list of strings")
        missing_fields = set(required) - set(properties)
        if missing_fields:
            raise ValueError(f"required parameters are not declared: {sorted(missing_fields)}")
        undeclared_locations = set(self.argument_locations) - set(properties)
        if undeclared_locations:
            raise ValueError(
                f"argument locations reference undeclared parameters: {sorted(undeclared_locations)}"
            )
        missing_locations = set(properties) - set(self.argument_locations)
        if missing_locations:
            raise ValueError(
                f"parameters must declare an argument location: {sorted(missing_locations)}"
            )
        return self


class ToolsConfig(BaseModel):
    """tools.yaml 的根配置模型。"""

    tools: list[ToolConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def tool_names_must_be_unique(self) -> "ToolsConfig":
        """确认所有工具名称唯一。

        逻辑规划：
        1. 收集所有工具名称。
        2. 对比名称数量和去重后数量。
        3. 出现重复名称时阻止注册阶段产生不确定的工具映射。
        """

        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")
        return self


@lru_cache(maxsize=1)
def get_tools_config() -> ToolsConfig:
    """读取、解析并缓存 tools.yaml。

    逻辑规划：
    1. 确认配置文件存在，缺失时直接报出可定位错误。
    2. 使用 safe_load 解析 YAML，避免执行任意 YAML 对象。
    3. 交给 Pydantic 完成字段类型、范围和交叉字段校验。
    """

    if not TOOLS_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Tools configuration not found: {TOOLS_CONFIG_PATH}")

    with TOOLS_CONFIG_PATH.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    return ToolsConfig.model_validate(raw_config)
