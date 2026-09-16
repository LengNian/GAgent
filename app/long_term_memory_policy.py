"""长期记忆属性的单值与多值策略。"""


# 这些属性代表当前唯一状态，新值会替代旧值；其他属性允许多条并存。
SINGLE_VALUE_LONG_TERM_MEMORY_ATTRIBUTES = frozenset({"name", "residence"})


def is_single_value_long_term_memory_attribute(attribute: str) -> bool:
    """判断属性是否只允许一条当前有效记忆。"""

    return attribute in SINGLE_VALUE_LONG_TERM_MEMORY_ATTRIBUTES
