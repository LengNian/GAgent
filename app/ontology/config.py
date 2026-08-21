"""读取并校验 Ontology 与 Action 配置。"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


ONTOLOGY_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "actions.yaml"


class ObjectTypeConfig(BaseModel):
    """Ontology 中可被 Action 操作的对象类型。"""

    # 本体中可被操作的对象类型（如Device等）
    name: str = Field(pattern=r"^[A-Z][A-Za-z0-9]*$")
    description: str = Field(min_length=1)
    # 该对象的属性
    properties: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RelationTypeConfig(BaseModel):
    """Ontology 中两个对象类型之间的关系。"""

    name: str = Field(pattern=r"^[A-Z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    target_type: str = Field(min_length=1)
    cardinality: Literal["one-to-one", "one-to-many", "many-to-one", "many-to-many"]


class PreconditionConfig(BaseModel):
    """可由 Action Gateway 确定性执行的前置条件。"""

    type: Literal["validator"]
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    # 校验的输入字段
    field: str = Field(min_length=1)


class ActionConfig(BaseModel):
    """Ontology 中可被 Agent 请求的业务 Action 契约。"""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    target_type: str = Field(min_length=1)
    input_schema: dict[str, Any]
    preconditions: list[PreconditionConfig] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high", "critical"]
    requires_confirmation: bool
    executor: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")

    @model_validator(mode="after")
    def validate_input_schema(self) -> "ActionConfig":
        """校验 Action 输入 schema 的最小结构。

        逻辑规划：
        1. 确认输入 schema 声明为 object，避免 Action 参数退化为任意值。
        2. 确认 properties 是对象，保证参数可以被后续执行器解析。
        3. 确认 required 只引用已声明字段，避免模型生成无法执行的调用。
        4. 对高风险 Action 强制要求人工确认，避免配置遗漏造成直接副作用。
        """

        if self.input_schema.get("type") != "object":
            raise ValueError("action input_schema.type must be object")

        properties = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])

        if not isinstance(properties, dict):
            raise ValueError("action input_schema.properties must be an object")

        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            raise ValueError("action input_schema.required must be a list of strings")

        missing_fields = set(required) - set(properties)
        if missing_fields:
            raise ValueError(
                f"action required parameters are not declared: {sorted(missing_fields)}"
            )

        if self.risk_level in {"high", "critical"} and not self.requires_confirmation:
            raise ValueError("high-risk actions must require confirmation")

        return self


class OntologyConfig(BaseModel):
    """actions.yaml 的根配置模型。"""

    object_types: list[ObjectTypeConfig] = Field(default_factory=list)
    relations: list[RelationTypeConfig] = Field(default_factory=list)
    actions: list[ActionConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references_and_names(self) -> "OntologyConfig":
        """校验 Ontology 名称唯一且 Action 引用已注册对象。

        逻辑规划：
        1. 检查对象、关系和 Action 名称是否重复，避免注册表出现覆盖。
        2. 检查关系两端的对象类型是否存在。
        3. 检查 Action 目标对象类型是否存在。
        4. 发现引用错误时阻止配置加载，让服务在启动阶段暴露问题。
        """

        object_names = [item.name for item in self.object_types]
        relation_names = [item.name for item in self.relations]
        action_names = [item.name for item in self.actions]

        if len(object_names) != len(set(object_names)):
            raise ValueError("ontology object type names must be unique")
        if len(relation_names) != len(set(relation_names)):
            raise ValueError("ontology relation type names must be unique")
        if len(action_names) != len(set(action_names)):
            raise ValueError("ontology action names must be unique")

        known_objects = set(object_names)
        for relation in self.relations:
            if relation.source_type not in known_objects or relation.target_type not in known_objects:
                raise ValueError(f"relation references unknown object type: {relation.name}")
        for action in self.actions:
            if action.target_type not in known_objects:
                raise ValueError(f"action references unknown object type: {action.name}")
        return self


@lru_cache(maxsize=1)
def get_ontology_config() -> OntologyConfig:
    """读取并缓存 Ontology 配置。

    逻辑规划：
    1. 确认配置文件存在，缺失时抛出可定位错误。
    2. 使用 safe_load 解析 YAML，拒绝执行任意 YAML 对象。
    3. 使用 Pydantic 校验字段、风险规则和跨对象引用。
    """

    if not ONTOLOGY_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Ontology configuration not found: {ONTOLOGY_CONFIG_PATH}")

    with ONTOLOGY_CONFIG_PATH.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
        
    return OntologyConfig.model_validate(raw_config)
