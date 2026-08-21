"""Ontology 对象、关系和 Action 注册能力。"""

from .config import (
    ActionConfig,
    OntologyConfig,
    ObjectTypeConfig,
    PreconditionConfig,
    RelationTypeConfig,
    get_ontology_config,
)
from .registry import ActionRegistry, get_action_registry

__all__ = [
    "ActionConfig",
    "ActionRegistry",
    "OntologyConfig",
    "ObjectTypeConfig",
    "PreconditionConfig",
    "RelationTypeConfig",
    "get_action_registry",
    "get_ontology_config",
]
