"""gateways.yaml 的解析与启动校验。

逻辑规划：
1. [模型定义] 用 Pydantic 声明中台与 API 的配置契约，字段校验失败即拒绝启动，
   与 app/startup.py 的"配置缺失阻止启动"原则保持一致。
2. [env 引用] base_url、凭证等敏感值统一写 `env:VAR_NAME` 形式，
   解析时从环境变量取值；变量缺失直接抛错，绝不允许明文凭证进配置文件。
3. [跨字段校验] argument_locations 必须与 input_schema.properties 对齐、
   required 必须是 properties 子集、登录取值路径字段必须存在，
   防止配置错误延迟到运行时才暴露。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class EnvReferenceError(ValueError):
    """配置引用的环境变量不存在。"""


def resolve_env_reference(value: str, *, context: str) -> str:
    """解析 `env:VAR_NAME` 形式的配置值；普通值原样返回。

    Args:
        value: 配置文件中的原始字符串值。
        context: 报错时说明是哪个配置项，便于定位。
    Returns:
        解析后的实际值。
    Raises:
        EnvReferenceError: 值声明为 env 引用但环境变量不存在或为空。
    """

    if not value.startswith("env:"):
        return value
    variable_name = value[4:]
    resolved = os.environ.get(variable_name, "")
    if not resolved.strip():
        raise EnvReferenceError(f"环境变量未设置或为空: {variable_name} (配置项: {context})")
    return resolved


class AuthConfig(BaseModel):
    """"登录换 token"型中台的鉴权配置。"""

    model_config = ConfigDict(extra="forbid")

    type: Literal["login_token"]
    login_endpoint: str = Field(min_length=1)
    refresh_endpoint: str | None = None
    username_env: str = Field(min_length=1)
    password_env: str = Field(min_length=1)
    token_field: str = Field(default="data.access_token")
    refresh_token_field: str = Field(default="data.refresh_token")
    expires_in_field: str | None = Field(default="data.expires_in")


class ApiConfig(BaseModel):
    """单条 API 的执行映射：模型看到的工具契约 + 到中台的 HTTP 细节。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    method: Literal["GET", "POST", "PUT", "DELETE"]
    endpoint: str = Field(min_length=1)
    timeout_seconds: float = Field(default=10, gt=0)
    retry_attempts: int = Field(default=2, ge=0)
    retry_delay_seconds: float = Field(default=0.5, ge=0)
    risk_level: Literal["low", "medium", "high", "critical"] = "low"
    requires_confirmation: bool = False
    argument_locations: dict[str, Literal["path", "query", "body"]] = Field(default_factory=dict)
    input_schema: dict[str, Any]
    error_messages: dict[str, str] = Field(default_factory=dict)
    # 返回给 Agent 前应用的投影器名字，须与 mcp_gateway.projections.PROJECTIONS 的键一致；
    # 用 Literal 而非 str，是为了让“配了名字但没注册投影函数”在配置加载期就报错。
    result_projection: Literal["topology_graph", "device_series", "metric_series"] | None = None
    # 请求中台之前必须通过的参数校验器名字，须与 mcp_gateway.argument_guards.GUARDS 的键一致。
    # 与 result_projection 对称，同样用 Literal 换取值拼错在启动期暴露。
    argument_guard: Literal["time_window"] | None = None

    @model_validator(mode="after")
    def locations_and_schema_must_align(self) -> "ApiConfig":
        """参数位置表与 schema 的属性必须一一对应，避免运行期静默丢参。"""

        properties = self.input_schema.get("properties", {})
        if not isinstance(properties, dict):
            raise ValueError(f"API {self.name}: input_schema.properties 必须是映射")
        required = self.input_schema.get("required", [])
        if not isinstance(required, list):
            raise ValueError(f"API {self.name}: input_schema.required 必须是列表")
        # properties 允许为空（无参接口，如全局拓扑图）；有参时位置表必须对齐
        if properties and set(self.argument_locations) != set(properties):
            raise ValueError(
                f"API {self.name}: argument_locations 键 {sorted(self.argument_locations)} "
                f"必须与 input_schema.properties 键 {sorted(properties)} 一致"
            )
        unknown_required = set(required) - set(properties)
        if unknown_required:
            raise ValueError(f"API {self.name}: required 引用了未声明的参数 {sorted(unknown_required)}")
        if self.method == "GET" and "body" in self.argument_locations.values():
            raise ValueError(f"API {self.name}: GET 请求不能有 body 参数")
        if self.risk_level in {"high", "critical"} and not self.requires_confirmation:
            raise ValueError(f"API {self.name}: 高风险操作必须要求人工确认")
        # 校验器依赖具体参数存在；配了却没声明参数属于配置错，不能留到运行期才发现
        if self.argument_guard == "time_window" and not {"start", "end"} & set(properties):
            raise ValueError(f"API {self.name}: argument_guard=time_window 要求声明 start 或 end 参数")
        return self


class PlatformConfig(BaseModel):
    """单个中台的连接与工具声明。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    base_url_env: str = Field(min_length=1)
    auth: AuthConfig
    apis: list[ApiConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def tool_names_must_be_unique(self) -> "PlatformConfig":
        """工具名在平台内必须唯一；跨平台靠 name 前缀隔离。"""

        names = [api.name for api in self.apis]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"中台 {self.name} 存在重复工具名: {sorted(duplicates)}")
        return self


class GatewaySettings(BaseModel):
    """gateways.yaml 的解析结果，网关所有模块的唯一配置来源。"""

    model_config = ConfigDict(extra="forbid")

    platforms: list[PlatformConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def platform_names_must_be_unique(self) -> "GatewaySettings":
        """中台名即工具名前缀，重名会导致工具路由歧义。"""

        names = [platform.name for platform in self.platforms]
        duplicates = {name for name in names if names.count(name) > 1}
        if duplicates:
            raise ValueError(f"存在重复的中台名: {sorted(duplicates)}")
        return self

    def resolve_base_url(self, platform: PlatformConfig) -> str:
        """解析中台 base_url（env 引用）。"""

        return resolve_env_reference(platform.base_url_env, context=f"{platform.name}.base_url")


def load_gateway_settings(config_path: str | Path) -> GatewaySettings:
    """读取并校验网关配置。

    Args:
        config_path: gateways.yaml 路径。
    Returns:
        校验通过的 GatewaySettings。
    Raises:
        FileNotFoundError: 配置文件不存在。
        ValidationError: 配置不符合契约。
    """

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"网关配置不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"网关配置必须是 YAML 映射: {path}")
    return GatewaySettings.model_validate(raw)
