"""LangGraph Agent 和双 Agent Supervisor 编排图的构造逻辑。"""

import asyncio
import json
import logging
from typing import Any, Literal, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field, ValidationError

from app.action_gateway import ActionResult
from app.prompt_loader import get_agent_prompt, get_report_prompt
from app.agent_manifest import get_agent_manifest
from app.settings import Settings, get_settings
from app.tools.registry import build_tools_for_agent


logger = logging.getLogger(__name__)


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


# 把ToolMessage里的内容统一解析为Python dict    
def _parse_tool_result(content: object) -> dict[str, Any] | None:
    """将工具消息内容转换为结构化结果。

    逻辑规划：
    1. 接受 Gateway 常见的 JSON 字符串结果。
    2. 兼容 LangChain 将模型内容表示为文本块列表的形式。
    3. 对已经解析的字典直接使用，其他结构视为不可识别。
    4. JSON 无法解析或结果不是对象时返回 None，由上层继续正常流程。
    """

    # print("**********", content)

    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        text_parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        content = "".join(text_parts)
    if not isinstance(content, str):
        return None
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return None
    return result if isinstance(result, dict) else None


# 判断_parse_tool_result输出的内容是不是合法的ActionResult
def _action_result_from_tool_content(content: object) -> ActionResult | None:
    """解析并校验 ToolMessage 中的 ActionResult。

    逻辑规划：
    1. 将 ToolMessage 内容解析为 JSON 对象。
    2. 使用 ActionResult 校验结构和字段类型，拒绝不完整结果。
    3. 解析失败时返回 None，由 Report Node 使用保守提示。
    """

    raw_result = _parse_tool_result(content)
    if raw_result is None:
        return None

    try:
        return ActionResult.model_validate(raw_result)
    except ValidationError:
        return None


def _failure_report_message(result: ActionResult) -> str:
    """将失败 ActionResult 转换为确定性的用户提示。

    逻辑规划：
    1. 对用户可修复的具体错误码给出明确下一步。
    2. 对外部服务和响应失败按错误类别说明结果不可信。
    3. 对未知失败返回通用安全提示，不泄露内部实现细节。
    """

    if result.error_code == "invalid_ipv4":
        return "设备 IPv4 地址格式不正确，请提供类似 192.168.1.111 的地址。"
    if result.error_code == "missing_required_argument":
        fields = result.details.get("fields", [])
        field_text = "、".join(str(field) for field in fields)
        return f"缺少必要查询参数：{field_text or '未知字段'}。"
    if result.error_code == "upstream_not_found":
        return "未找到该 IP 对应的设备，请确认设备地址后重试。"
    if result.error_type == "transport":
        return "设备查询服务暂时不可用，请稍后重试。"
    if result.error_type == "response":
        return "设备查询服务返回异常，未能生成可信的查询结果。"
    if result.error_type == "authorization":
        return "当前请求未获执行授权，无法查询设备信息。"
    if result.error_type == "validation":
        return "查询参数不符合要求，请检查后重试。"
    return "当前操作未能完成，系统已阻止不可信结果返回。"


async def _summarize_successful_action_result(
    result: ActionResult,
    model: BaseChatModel,
) -> str:
    """使用独立报告模型总结已确认成功的 ActionResult。

    逻辑规划：
    1. 仅将 Action 名称和 data 作为事实输入交给 Report Prompt。
    2. 不绑定任何工具，阻止报告模型继续执行 Action。
    3. 模型异常或返回空文本时使用不回显原始数据的安全提示。
    """

    report_input = json.dumps(
        {"action_name": result.action_name, "data": result.data},
        ensure_ascii=False,
    )
    try:
        response = await model.ainvoke(
            [
                SystemMessage(content=get_report_prompt()),
                HumanMessage(content=report_input),
            ]
        )
    except Exception:
        logger.exception("Report model invocation failed for action %s", result.action_name)
        return "设备查询已完成，但结果摘要生成失败，请稍后重试。"

    summary = getattr(response, "content", "")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    return "设备查询已完成，但结果摘要生成失败，请稍后重试。"


# 倒着找最后一个ToolMessage，然后解析它的content
async def _report_message_for_messages(
    messages: list[Any],
    model: BaseChatModel,
) -> str:
    """从最新 ToolMessage 读取结果并生成最终报告。

    逻辑规划：
    1. 从尾部查找 ToolMessage，确保只消费 ToolNode 的实际输出。
    2. 失败结果走确定性文案，成功结果交给无工具的报告模型总结。
    3. 没有工具结果或结果无法解析时返回保守提示，不回显原始内容。
    """

    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            result = _action_result_from_tool_content(message.content)
            if result is None:
                return "工具返回结果异常，无法确认本次设备查询是否完成。"
            if not result.ok:
                return _failure_report_message(result)
            return await _summarize_successful_action_result(result, model)
    return "未收到工具执行结果，无法确认本次设备查询是否完成。"


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
    3. 对有工具的领域构建“领域模型 -> 工具 -> Report Node -> 结束”链路。
    4. 没有工具调用时结束当前领域图，保留普通对话 Agent 的行为。
    """

    tools = build_tools_for_agent(settings, agent_id)
    model_with_tools = model.bind_tools(tools) if tools else model

    async def call_model(state: MessagesState) -> dict[str, list[Any]]:
        """调用领域模型并追加一条模型消息。"""

        response = await model_with_tools.ainvoke(_messages_with_prompt(agent_id, state))
        return {"messages": [response]}

    async def report(state: MessagesState) -> dict[str, list[Any]]:
        """根据实际工具结果生成确定性最终报告。

        逻辑规划：
        1. 只读取 ToolNode 追加的 ToolMessage，不使用模型文本推断执行结果。
        2. 将结构化 ActionResult 转换为成功数据或安全失败说明。
        3. 追加最终 AIMessage 并结束子图，阻止工具调用后的模型二次生成。
        """
        report_message = await _report_message_for_messages(
            state.get("messages", []),
            model,
        )
        return {"messages": [AIMessage(content=report_message)]}

    graph = StateGraph(MessagesState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    if tools:
        graph.add_node("tools", ToolNode(tools))
        graph.add_node("report", report)
        graph.add_conditional_edges("call_model", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "report")
        graph.add_edge("report", END)
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
