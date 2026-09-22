"""Agent 模型消息组装工具。"""

from typing import Any

from langchain_core.messages import SystemMessage

from app.prompt_loader import get_agent_prompt
from app.agent.models import AgentGraphState, DomainGraphState


def messages_with_prompt(
    agent_id: str,
    state: AgentGraphState | DomainGraphState,
) -> list[Any]:
    """为指定 Agent 组装系统 Prompt、上下文和当前执行消息。"""

    return [
        SystemMessage(content=get_agent_prompt(agent_id)),
        *state.get("context_messages", []),
        *state.get("execution_messages", []),
    ]
