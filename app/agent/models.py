"""Agent 图共享的数据模型与异常。"""
from typing import Any, Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    """Supervisor 输出的结构化路由结果。"""
    target_agent: Literal["conversation_agent", "iot_agent"]
    intent: str = Field(min_length=1)
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    decision_summary: str = Field(min_length=1, max_length=160)

class AgentGraphState(TypedDict, total=False):
    """双 Agent 编排图共享的运行状态。"""
    context_messages: list[Any]
    execution_messages: Annotated[list[Any], add_messages]
    target_agent: str
    intent: str
    entities: dict[str, Any]
    confidence: float
    decision_summary: str

class DomainGraphState(TypedDict, total=False):
    """领域子图共享的运行状态。"""
    context_messages: list[Any]
    execution_messages: Annotated[list[Any], add_messages]
    approval_rejected: bool

class AgentExecutionLimitError(Exception):
    """Agent 执行超过 manifest 限制。"""
    def __init__(self, agent_id: str, limit_type: str, limit_value: float | int) -> None:
        self.agent_id = agent_id
        self.limit_type = limit_type
        self.limit_value = limit_value
        super().__init__(f"Agent {agent_id} exceeded {limit_type}: {limit_value}")

class SupervisorRoutingError(Exception):
    """Supervisor 未返回可用结构化路由结果。"""
