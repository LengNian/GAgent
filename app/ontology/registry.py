"""提供经过校验的 Ontology Action 查询。"""

from functools import lru_cache

from .config import ActionConfig, get_ontology_config


class ActionRegistry:
    """按名称查询已注册 Action 的只读注册表。"""

    def __init__(self, actions: list[ActionConfig]) -> None:
        """创建 Action 注册表。

        逻辑规划：
        1. 将已经由 OntologyConfig 校验过的 Action 转为名称索引。
        2. 再次拒绝重复名称，避免调用方绕过根配置模型时产生覆盖。
        3. 保存不可变语义配置；执行权限和具体执行在后续 Action Gateway 处理。
        """

        action_map = {action.name: action for action in actions}
        if len(action_map) != len(actions):
            raise ValueError("action names must be unique")
        self._actions = action_map

    def get(self, action_name: str) -> ActionConfig:
        """按名称返回 Action，不存在时抛出明确错误。"""

        try:
            return self._actions[action_name]
        except KeyError as error:
            raise KeyError(f"Action is not registered: {action_name}") from error

    def list(self) -> list[ActionConfig]:
        """返回当前注册的 Action 快照。"""

        return list(self._actions.values())


@lru_cache(maxsize=1)
def get_action_registry() -> ActionRegistry:
    """加载并缓存当前进程的 Action Registry。"""

    return ActionRegistry(get_ontology_config().actions)
