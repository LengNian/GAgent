"""LangGraph Agent 和双 Agent Supervisor 编排图的构造逻辑。"""

import asyncio
import json
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from app.prompt_loader import get_agent_prompt
from app.agent_manifest import get_agent_manifest
from app.settings import Settings, get_settings
from app.tools.registry import build_tools_for_agent


class RouteDecision(BaseModel):
    """Supervisor 输出的结构化路由结果。"""

    target_agent: Literal["conversation_agent", "iot_agent"]
    intent: str = Field(min_length=1)
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    decision_summary: str = Field(min_length=1, max_length=160)


class AgentGraphState(MessagesState, total=False):
    """双 Agent 编排图共享的运行状态。"""

    target_agent: str
    intent: str
    entities: dict[str, Any]
    confidence: float
    decision_summary: str


class DomainGraphState(MessagesState, total=False):
    """领域 Agent 图状态，额外保存确定性的降级结果。"""

    fallback_message: str


class AgentExecutionLimitError(Exception):
    """Agent 执行超过 manifest 声明的时间或步骤限制。"""

    def __init__(self, agent_id: str, limit_type: str, limit_value: float | int) -> None:
        self.agent_id = agent_id
        self.limit_type = limit_type
        self.limit_value = limit_value
        super().__init__(f"Agent {agent_id} exceeded {limit_type}: {limit_value}")


def _build_model(settings: Settings) -> BaseChatModel:
    """根据已校验配置创建聊天模型。"""

    model_kwargs: dict[str, Any] = {
        "api_key": settings.llm_api_key.get_secret_value(),
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "timeout": settings.llm_timeout_seconds,
    }
    if settings.llm_base_url:
        model_kwargs["base_url"] = settings.llm_base_url
    return ChatOpenAI(**model_kwargs)


def _messages_with_prompt(agent_id: str, state: MessagesState) -> list[Any]:
    """为指定 Agent 组装系统 Prompt 和当前消息。"""

    return [SystemMessage(content=get_agent_prompt(agent_id)), *state["messages"]]


def _fallback_message_for_tool_result(content: object, agent_id: str) -> str | None:
    """按 Agent、Action 和错误类型生成安全降级文案。"""

    if not isinstance(content, str):
        return None
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(result, dict) or result.get("ok") is not False:
        return None

    action_name = result.get("action_name")
    error_type = result.get("error_type")

    if action_name == "query_device_by_ip":
        return "设备查询未完成，外部设备服务暂时不可用，请稍后重试。"
    if error_type in {"network", "timeout", "server"}:
        return f"{agent_id} 依赖的外部服务暂时不可用，请稍后重试。"
    return f"{agent_id} 未能完成当前操作，请检查输入后重试。"


async def _invoke_domain_graph(
    *,
    agent_id: str,
    graph: Any,
    messages: list[Any],
) -> dict[str, Any]:
    """按 Agent manifest 限制执行领域子图。

    逻辑规划：
    1. 读取 Agent manifest 中的总超时与最大图步骤数。
    2. 在超时范围内调用领域图，并将最大步骤数传给 LangGraph。
    3. 将框架超步数和 asyncio 超时转换为稳定的业务异常。
    """

    runtime = get_agent_manifest(agent_id).runtime
    try:
        async with asyncio.timeout(runtime.timeout_seconds):
            return await graph.ainvoke(
                {"messages": messages},
                config={"recursion_limit": runtime.max_steps},
            )
    except TimeoutError as error:
        raise AgentExecutionLimitError(
            agent_id,
            "timeout_seconds",
            runtime.timeout_seconds,
        ) from error
    except GraphRecursionError as error:
        raise AgentExecutionLimitError(
            agent_id,
            "max_steps",
            runtime.max_steps,
        ) from error


def _create_domain_agent(
    *,
    agent_id: str,
    model: BaseChatModel,
    settings: Settings,
):
    """创建单个领域 Agent 图，供直接调用或 Supervisor 节点复用。

    逻辑规划：
    1. 根据 manifest 裁剪该 Agent 的 Action 工具集合。
    2. 将 Agent 专属 Prompt 注入模型上下文。
    3. 构建“领域模型 -> 工具 -> 领域模型”的循环。
    4. 没有工具调用时结束当前领域图。
    """

    tools = build_tools_for_agent(settings, agent_id)
    model_with_tools = model.bind_tools(tools) if tools else model

    async def call_model(state: MessagesState) -> dict[str, list[Any]]:
        """调用领域模型并追加一条模型消息。"""

        response = await model_with_tools.ainvoke(_messages_with_prompt(agent_id, state))
        return {"messages": [response]}

    async def inspect_tool_result(state: DomainGraphState) -> dict[str, str]:
        """检查最近一次工具结果，失败时准备进入 fallback。"""

        if not state["messages"]:
            return {}
        fallback_message = _fallback_message_for_tool_result(
            state["messages"][-1].content,
            agent_id,
        )
        return {"fallback_message": fallback_message} if fallback_message else {}

    async def call_fallback(state: DomainGraphState) -> dict[str, list[Any]]:
        """返回不编造业务数据的安全降级回答。"""

        return {
            "messages": [
                AIMessage(
                    content=state.get(
                        "fallback_message",
                        "当前请求未能完成，请稍后重试。",
                    )
                )
            ]
        }

    def route_after_tool(state: DomainGraphState) -> str:
        """根据工具结果选择继续生成或进入 fallback。"""

        return "fallback" if state.get("fallback_message") else "call_model"

    graph = StateGraph(DomainGraphState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_node("inspect_tool_result", inspect_tool_result)
        graph.add_node("fallback", call_fallback)
        graph.add_conditional_edges("call_model", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "inspect_tool_result")
        graph.add_conditional_edges(
            "inspect_tool_result",
            route_after_tool,
            {"call_model": "call_model", "fallback": "fallback"},
        )
        graph.add_edge("fallback", END)
    else:
        graph.add_edge("call_model", END)
    return graph.compile()


def _create_orchestrated_agent(
    *,
    model: BaseChatModel,
    settings: Settings,
):
    """创建 Supervisor -> Conversation/IoT 的 LangGraph 编排图。

    逻辑规划：
    1. Supervisor 使用结构化输出识别意图、实体、置信度和目标 Agent。
    2. 条件边只允许路由到 Conversation Agent 或 IoT Agent。
    3. Conversation Agent 不绑定外部 Action；IoT Agent 只绑定自己的 allowlist。
    4. 领域 Agent 完成后结束本轮图执行，最终消息由 API 层流式返回。
    """

    # GLM 的 OpenAI 兼容接口对 response_format 的支持不完整，会将 JSON Schema
    # 当作普通文本返回；函数调用能保证路由结果以工具参数形式返回。
    supervisor_model = model.with_structured_output(RouteDecision, method="function_calling")
    conversation_graph = _create_domain_agent(
        agent_id="conversation_agent",
        model=model,
        settings=settings,
    )
    iot_graph = _create_domain_agent(
        agent_id="iot_agent",
        model=model,
        settings=settings,
    )

    async def call_supervisor(state: AgentGraphState) -> dict[str, object]:
        """调用 Supervisor 并保存结构化路由字段，不把路由结果写入对话消息。"""

        runtime = get_agent_manifest("supervisor").runtime
        try:
            async with asyncio.timeout(runtime.timeout_seconds):
                decision = await supervisor_model.ainvoke(_messages_with_prompt("supervisor", state))
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

    def route_to_agent(state: AgentGraphState) -> str:
        """根据 Supervisor 结果选择唯一的领域 Agent 节点。"""

        target_agent = state.get("target_agent")
        if target_agent not in {"conversation_agent", "iot_agent"}:
            raise ValueError(f"Supervisor returned unsupported target agent: {target_agent}")
        return target_agent

    async def call_conversation(state: AgentGraphState) -> dict[str, list[Any]]:
        """执行 Conversation Agent 子图。"""

        result = await _invoke_domain_graph(
            agent_id="conversation_agent",
            graph=conversation_graph,
            messages=state["messages"],
        )
        return {"messages": result["messages"][-1:]}

    async def call_iot(state: AgentGraphState) -> dict[str, list[Any]]:
        """执行 IoT Agent 子图。"""

        result = await _invoke_domain_graph(
            agent_id="iot_agent",
            graph=iot_graph,
            messages=state["messages"],
        )
        return {"messages": result["messages"][-1:]}

    graph = StateGraph(AgentGraphState)
    graph.add_node("supervisor", call_supervisor)
    graph.add_node("conversation_agent", call_conversation)
    graph.add_node("iot_agent", call_iot)

    
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
    return graph.compile()


def create_agent(
    *,
    agent_id: str | None = None,
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
):
    """创建编排图或指定的单领域 Agent 图。

    Args:
        agent_id: 传入时创建指定领域 Agent；不传入时创建 Supervisor 编排图。
        model: 可选的注入模型，便于测试和替换模型实现。
        settings: 可选的已校验配置；未传入时加载默认配置。
    Returns:
        已编译的 LangGraph 图。
    """

    resolved_settings = settings or get_settings()
    chat_model = model or _build_model(resolved_settings)
    if agent_id:
        return _create_domain_agent(
            agent_id=agent_id,
            model=chat_model,
            settings=resolved_settings,
        )
    return _create_orchestrated_agent(model=chat_model, settings=resolved_settings)
