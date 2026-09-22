"""LangGraph Agent 工厂与 Supervisor 编排图。"""

import asyncio
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from app.agent.domain_graph import create_domain_graph, invoke_domain_graph
from app.agent.message_builder import messages_with_prompt
from app.agent.model_runtime import build_model
from app.agent.models import (
    AgentExecutionLimitError,
    AgentGraphState,
    RouteDecision,
)
from app.agent.routing import invoke_route_decision
from app.agent_manifest import get_agent_manifest
from app.settings import Settings, get_settings


async def _create_orchestrated_agent(
    *,
    model: BaseChatModel,
    settings: Settings,
    checkpointer: Any | None = None,
):
    """创建 Supervisor -> Conversation/IoT 的 LangGraph 编排图。

    Supervisor 只负责路由。会话领域图不触发 MCP 工具发现，IoT 领域图
    仅在真正路由到 IoT 后才发现工具，避免普通对话受到外部网关影响。
    """

    supervisor_model = model.with_structured_output(
        RouteDecision,
        method="function_calling",
    )
    conversation_graph = await create_domain_graph(
        agent_id="conversation_agent",
        model=model,
        settings=settings,
    )

    async def call_conversation_agent(
        state: AgentGraphState,
        config: RunnableConfig,
    ) -> dict[str, list[Any]]:
        """在 manifest 限制内执行会话领域子图。"""

        result = await invoke_domain_graph(
            agent_id="conversation_agent",
            graph=conversation_graph,
            messages=state.get("context_messages", []),
            config=config,
        )
        return {"execution_messages": result.get("execution_messages", [])}

    async def call_supervisor(state: AgentGraphState) -> dict[str, object]:
        """调用 Supervisor 并保存结构化路由字段。"""

        runtime = get_agent_manifest("supervisor").runtime
        try:
            async with asyncio.timeout(runtime.timeout_seconds):
                decision = await invoke_route_decision(
                    supervisor_model,
                    messages_with_prompt("supervisor", state),
                )
        except TimeoutError as error:
            raise AgentExecutionLimitError(
                "supervisor",
                "timeout_seconds",
                runtime.timeout_seconds,
            ) from error
        return {
            "target_agent": decision.target_agent,
            "intent": decision.intent,
            "entities": decision.entities,
            "confidence": decision.confidence,
            "decision_summary": decision.decision_summary,
        }

    async def call_iot_agent(
        state: AgentGraphState,
        config: RunnableConfig,
    ) -> dict[str, list[Any]]:
        """Supervisor 路由成功后创建并执行 IoT 领域子图。"""

        iot_graph = await create_domain_graph(
            agent_id="iot_agent",
            model=model,
            settings=settings,
            checkpointer=checkpointer,
        )
        result = await invoke_domain_graph(
            agent_id="iot_agent",
            graph=iot_graph,
            messages=state.get("context_messages", []),
            config=config,
        )
        return {"execution_messages": result.get("execution_messages", [])}

    def route_to_agent(state: AgentGraphState) -> str:
        """拒绝 Supervisor 返回的未知目标，防止图进入未定义节点。"""

        target_agent = state.get("target_agent")
        if target_agent not in {"conversation_agent", "iot_agent"}:
            raise ValueError(f"Supervisor returned unsupported target agent: {target_agent}")
        return target_agent

    graph = StateGraph(AgentGraphState)
    graph.add_node("supervisor", call_supervisor)
    graph.add_node("conversation_agent", call_conversation_agent)
    graph.add_node("iot_agent", call_iot_agent)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "conversation_agent": "conversation_agent",
            "iot_agent": "iot_agent",
        },
    )
    graph.add_edge("conversation_agent", END)
    graph.add_edge("iot_agent", END)
    return graph.compile(checkpointer=checkpointer)


async def create_agent(
    *,
    agent_id: str | None = None,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
    checkpointer: Any | None = None,
):
    """创建指定领域图或默认 Supervisor 编排图。"""

    resolved_settings = settings or get_settings()
    chat_model = model or build_model(resolved_settings)
    if agent_id:
        return await create_domain_graph(
            agent_id=agent_id,
            model=chat_model,
            settings=resolved_settings,
            checkpointer=checkpointer,
        )
    return await _create_orchestrated_agent(
        model=chat_model,
        settings=resolved_settings,
        checkpointer=checkpointer,
    )
