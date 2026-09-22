"""LangGraph Agent 和双 Agent Supervisor 编排图的构造逻辑。"""

import asyncio
import json
import logging
from typing import Annotated, Any, Literal, TypedDict
from uuid import UUID

from anyio import to_thread
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError

from app.db.repositories import task_state_repository
from app.action_result import ActionResult
from app.prompt_loader import get_agent_prompt, get_report_prompt
from app.agent_manifest import get_agent_manifest
from app.settings import Settings, get_settings
from app.tools.registry import build_tools_for_agent


logger = logging.getLogger(__name__)

# Report 模型只需要结果摘要所需的数据，不应接收完整的长时间序列。
_REPORT_DATA_MAX_CHARS = 81920
ALLOWED_EMOTIONS = frozenset({"撒娇", "非常高兴", "非常生气", "悲伤", "困惑", "钦佩"})


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
    """可复用领域子图的运行状态。"""

    context_messages: list[Any]
    execution_messages: Annotated[list[Any], add_messages]
    approval_rejected: bool


class AgentExecutionLimitError(Exception):
    """Agent 执行超过 manifest 声明的时间或步骤限制。"""

    def __init__(self, agent_id: str, limit_type: str, limit_value: float | int) -> None:
        self.agent_id = agent_id
        self.limit_type = limit_type
        self.limit_value = limit_value
        super().__init__(f"Agent {agent_id} exceeded {limit_type}: {limit_value}")


class SupervisorRoutingError(Exception):
    """Supervisor 未返回可用结构化路由结果。"""


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


async def _invoke_route_decision(
    supervisor_model: Any,
    messages: list[Any],
) -> RouteDecision:
    """调用 Supervisor 并校验结构化路由结果，空结果最多重试一次。

    逻辑规划：
    1. 调用支持 function calling 的结构化模型。
    2. 只有实际得到 RouteDecision 才允许继续路由，拒绝 None 或其他返回类型。
    3. 空或非法结果重试一次；仍失败时抛出明确异常，禁止猜测目标 Agent。
    """

    for attempt in range(2):
        try:
            decision = await supervisor_model.ainvoke(messages)
        except Exception:
            if attempt == 1:
                raise
            logger.warning("supervisor_invocation_failed_retrying", exc_info=True)
            continue
        if isinstance(decision, RouteDecision):
            return decision
        logger.warning(
            "supervisor_invalid_decision attempt=%s result_type=%s",
            attempt + 1,
            type(decision).__name__,
        )

    raise SupervisorRoutingError("Supervisor 未返回有效路由结果")


def _messages_with_prompt(agent_id: str, state: AgentGraphState | DomainGraphState) -> list[Any]:
    """为指定 Agent 组装系统 Prompt 和当前消息。"""

    context_messages = state.get("context_messages", [])
    execution_messages = state.get("execution_messages", [])
    model_messages = [
        SystemMessage(content=get_agent_prompt(agent_id)),
        *context_messages,
        *execution_messages,
    ]
    return model_messages


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


def _fallback_report_emotion(result: ActionResult) -> str:
    """根据已校验的工具结果选择保守的 IoT 播报情绪。"""

    if result.ok:
        return "非常高兴"
    if result.error_type in {"transport", "response"}:
        return "悲伤"
    if result.error_code in {"invalid_ipv4", "missing_required_argument"}:
        return "困惑"
    return "非常生气"


def _parse_report_response(content: object, fallback_emotion: str) -> tuple[str, str]:
    """解析 Report 的可选结构化输出，拒绝未授权情绪和值类型。"""

    if isinstance(content, str):
        text = content.strip()
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return (text, fallback_emotion) if text else ("", fallback_emotion)
        if isinstance(parsed, dict):
            reply = parsed.get("reply")
            emotion = parsed.get("emotion")
            if isinstance(reply, str) and reply.strip():
                validated = emotion if isinstance(emotion, str) and emotion in ALLOWED_EMOTIONS else fallback_emotion
                return reply.strip(), validated
        return text, fallback_emotion
    return "", fallback_emotion


def _compact_report_value(value: Any, remaining_chars: int) -> tuple[Any, bool]:
    """在交给 Report 模型前限制工具结果体积。

    逻辑规划：
    1. 原始工具结果只读处理，不修改 ToolMessage 或数据库中的数据。
    2. 对列表保留首尾样本，适合时间序列同时保留起点和最近状态。
    3. 对对象递归裁剪；无法继续结构化裁剪时使用明确标记，避免请求超过模型窗口。
    """

    if remaining_chars <= 0:
        return "[结果过大，已省略]", True
    if isinstance(value, dict):
        compacted: dict[str, Any] = {}
        truncated = False
        for key, item in value.items():
            compacted_item, item_truncated = _compact_report_value(item, remaining_chars)
            compacted[str(key)] = compacted_item
            truncated = truncated or item_truncated
            encoded_size = len(json.dumps(compacted, ensure_ascii=False))
            if encoded_size > remaining_chars:
                compacted.pop(str(key), None)
                compacted["_truncated"] = "结果过大，部分字段已省略"
                return compacted, True
        return compacted, truncated
    if isinstance(value, list):
        if len(value) <= 20:
            compacted_items: list[Any] = []
            truncated = False
            for item in value:
                compacted_item, item_truncated = _compact_report_value(item, remaining_chars)
                compacted_items.append(compacted_item)
                truncated = truncated or item_truncated
                if len(json.dumps(compacted_items, ensure_ascii=False)) > remaining_chars:
                    compacted_items.pop()
                    compacted_items.append("[后续数据已省略]")
                    return compacted_items, True
            return compacted_items, truncated
        head = value[:10]
        tail = value[-10:]
        return [*head, "[中间数据已省略]", *tail], True
    if isinstance(value, str) and len(value) > remaining_chars:
        return value[: max(0, remaining_chars - 20)] + "...[已省略]", True
    return value, False


def _report_payload(result: ActionResult, max_chars: int = _REPORT_DATA_MAX_CHARS) -> dict[str, Any]:
    """构造有大小上限的单个工具结果报告输入。"""

    data, truncated = _compact_report_value(result.data if result.ok else None, max_chars)
    payload = {
        "ok": result.ok,
        "action_name": result.action_name,
        "data": data,
        "error_code": result.error_code if not result.ok else None,
        "error_type": result.error_type if not result.ok else None,
        "message": result.message if not result.ok else None,
        "details": result.details if not result.ok else {},
    }
    if truncated:
        payload["data_notice"] = "工具结果过大，报告仅使用首尾代表性数据；如需完整数据请缩小查询范围。"
    if len(json.dumps(payload, ensure_ascii=False)) > max_chars + 1200:
        payload["data"] = "[结果过大，已省略]"
        payload["data_notice"] = "工具结果过大，报告未使用完整数据；请缩小查询范围。"
    return payload


async def _summarize_successful_action_result(
    result: ActionResult,
    model: BaseChatModel,
) -> str:
    """使用独立报告模型总结成功或失败的 ActionResult。

    逻辑规划：
    1. 仅将 Action 状态、名称和安全结果字段交给 Report Prompt。
    2. 不绑定任何工具，阻止报告模型继续执行 Action。
    3. 模型异常或返回空文本时使用安全兜底提示。
    """

    report_input = json.dumps(_report_payload(result), ensure_ascii=False)
    fallback = (
        _failure_report_message(result)
        if not result.ok
        else "设备查询已完成，但结果摘要生成失败，请稍后重试。"
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
        return fallback

    summary, _ = _parse_report_response(getattr(response, "content", ""), _fallback_report_emotion(result))
    if summary:
        return summary
    return fallback


async def _summarize_successful_action_result_with_emotion(
    result: ActionResult, model: BaseChatModel
) -> tuple[str, str]:
    """生成 IoT 报告文本及经过白名单校验的情绪。"""

    fallback = _failure_report_message(result) if not result.ok else "设备查询已完成，但结果摘要生成失败，请稍后重试。"
    try:
        response = await model.ainvoke([
            SystemMessage(content=get_report_prompt()),
            HumanMessage(content=json.dumps(_report_payload(result), ensure_ascii=False)),
        ])
    except Exception:
        logger.exception("Report model invocation failed for action %s", result.action_name)
        return fallback, _fallback_report_emotion(result)
    text, emotion = _parse_report_response(getattr(response, "content", ""), _fallback_report_emotion(result))
    return (text or fallback), emotion


async def _summarize_action_results_with_emotion(
    results: list[ActionResult], model: BaseChatModel
) -> tuple[str, str]:
    """汇总多个工具结果并提取结构化情绪。"""

    fallback_result = next((result for result in results if not result.ok), None)
    fallback = (
        _failure_report_message(fallback_result)
        if fallback_result is not None
        else "设备查询已完成，但结果摘要生成失败，请稍后重试。"
    )
    fallback_emotion = _fallback_report_emotion(fallback_result) if fallback_result else "非常高兴"
    if len(results) == 1:
        return await _summarize_successful_action_result_with_emotion(results[0], model)
    try:
        response = await model.ainvoke([
            SystemMessage(content=get_report_prompt()),
            HumanMessage(content=json.dumps({"results": [_report_payload(result, max(2000, _REPORT_DATA_MAX_CHARS // len(results))) for result in results]}, ensure_ascii=False)),
        ])
    except Exception:
        logger.exception("Report model invocation failed for multiple actions")
        return fallback, fallback_emotion
    text, emotion = _parse_report_response(getattr(response, "content", ""), fallback_emotion)
    return (text or fallback), emotion


async def _report_result_for_messages(messages: list[Any], model: BaseChatModel) -> tuple[str, str]:
    """读取工具结果并返回最终文本和情绪。"""

    results: list[ActionResult] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            result = _action_result_from_tool_content(message.content)
            if result is None:
                return "工具返回结果异常，无法确认本次设备查询是否完成。", "困惑"
            results.append(result)
    if not results:
        return "未收到工具执行结果，无法确认本次设备查询是否完成。", "困惑"
    return await _summarize_action_results_with_emotion(results, model)


async def _invoke_domain_graph(
    *,
    agent_id: str,
    graph: Any,
    messages: list[Any],
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    """按 Agent manifest 限制执行领域子图。

    逻辑规划：
    1. 读取 Agent manifest 中的总超时与最大图步骤数。
    2. 在超时范围内调用领域图，并将最大步骤数传给 LangGraph。
    3. 将框架超步数和 asyncio 超时转换为稳定的业务异常。
    """

    runtime = get_agent_manifest(agent_id).runtime
    invocation_config = dict(config or {})
    invocation_config["recursion_limit"] = runtime.max_steps
    try:
        async with asyncio.timeout(runtime.timeout_seconds):
            return await graph.ainvoke(
                {
                    "context_messages": messages,
                    "execution_messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                },
                config=invocation_config,
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


async def _create_domain_agent(
    *,
    agent_id: str,
    model: BaseChatModel,
    settings: Settings,
    checkpointer: Any | None = None,
):
    """创建单个领域 Agent 图，供直接调用或 Supervisor 节点复用。

    逻辑规划：
    1. 根据 manifest 裁剪该 Agent 的 Action 工具集合。
    2. 将 Agent 专属 Prompt 注入模型上下文。
    3. 对有工具的领域构建“领域模型 -> 工具 -> 领域模型”循环，直到模型不再调用工具。
    4. 执行过工具后统一交给 Report Node 汇总；首次没有工具调用时直接结束。
    """

    tools = await build_tools_for_agent(settings, agent_id)
    model_with_tools = model.bind_tools(tools) if tools else model
    tool_policies = {
        tool.name: tool.metadata or {}
        for tool in tools
    }
    tools_by_name = {tool.name: tool for tool in tools}

    def _task_owner(config: RunnableConfig) -> tuple[UUID, str] | None:
        """从不可见运行配置读取审批状态的会话归属。"""

        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id") if isinstance(configurable, dict) else None
        user_id = configurable.get("user_id") if isinstance(configurable, dict) else None
        if not isinstance(thread_id, str) or not isinstance(user_id, str):
            return None
        try:
            return UUID(thread_id), user_id
        except ValueError:
            return None

    def _validated_pending_actions(state: DomainGraphState) -> list[dict[str, Any]]:
        """提取通过 MCP 审批策略和参数校验的待确认调用。"""

        messages = state.get("execution_messages", [])
        latest = messages[-1] if messages else None
        tool_calls = getattr(latest, "tool_calls", []) or []
        pending_actions: list[dict[str, Any]] = []
        
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_name = str(tool_call.get("name") or "")
            action_name = tool_name.rsplit(".", 1)[-1]
            arguments = tool_call.get("args")
            policy = tool_policies.get(tool_name)
            tool = tools_by_name.get(tool_name)
            if not action_name or not isinstance(arguments, dict) or not policy or tool is None:
                continue
            try:
                # MCP 工具 schema 是审批前唯一的参数契约。
                validated_arguments = tool.args_schema.model_validate(arguments).model_dump(
                    exclude_none=True
                )
            except (ValidationError, ValueError):
                continue
            if policy.get("requires_confirmation"):
                pending_actions.append(
                    {
                        "action_name": action_name,
                        "arguments": validated_arguments,
                        "risk_level": policy.get("risk_level", "low"),
                        "description": tool.description,
                    }
                )
        return pending_actions

    async def _finish_task(config: RunnableConfig, status: str) -> None:
        """在最终报告生成后清除待确认 Action。"""

        owner = _task_owner(config)
        if owner is None:
            return
        thread_id, user_id = owner
        await to_thread.run_sync(task_state_repository.finish_thread_task, thread_id, user_id, status)

    async def call_model(state: DomainGraphState) -> dict[str, list[Any]]:
        """调用领域模型并追加一条模型消息。"""

        # =========================================================================
        # [逻辑规划] 多轮工具调用决策
        # =========================================================================
        # 1. 依据当前 execution_messages 调用领域模型，首次调用由用户问题触发。
        # 2. 已存在 ToolMessage 时，模型只负责决定是否发起下一次工具调用。
        # 3. 为后续规划调用打上内部 tag，SSE 层不展示中间文本，最终结果交给 Report。
        # =========================================================================
        # 工具返回后的模型调用只负责决定下一步工具或结束；最终用户可见文本
        # 始终由 Report Node 生成，因此通过 tag 让 SSE 层跳过其中间规划文本。
        has_tool_result = any(
            isinstance(message, ToolMessage)
            for message in state.get("execution_messages", [])
        )
        config = {"tags": ["tool_planning"]} if has_tool_result else None
        response = await model_with_tools.ainvoke(
            _messages_with_prompt(agent_id, state),
            config=config,
        )
        return {"execution_messages": [response]}

    async def report(state: DomainGraphState, config: RunnableConfig) -> dict[str, list[Any]]:
        """根据实际工具结果生成确定性最终报告。

        逻辑规划：
        1. 只读取 ToolNode 追加的 ToolMessage，不使用模型文本推断执行结果。
        2. 将结构化 ActionResult 转换为成功数据或安全失败说明。
        3. 追加最终 AIMessage 并结束子图，阻止工具调用后的模型二次生成。
        """
        if state.get("approval_rejected"):
            await _finish_task(config, "rejected")
            return {"execution_messages": [AIMessage(content="操作已取消，未执行相关工具。", additional_kwargs={"emotion": "悲伤"})]}
        report_message, emotion = await _report_result_for_messages(
            state.get("execution_messages", []),
            model,
        )
        action_results = [
            _action_result_from_tool_content(message.content)
            for message in state.get("execution_messages", [])
            if isinstance(message, ToolMessage)
        ]
        completed = action_results and all(result is not None and result.ok for result in action_results)
        await _finish_task(config, "completed" if completed else "failed")
        return {"execution_messages": [AIMessage(content=report_message, additional_kwargs={"emotion": emotion})]}

    async def approval_gate(state: DomainGraphState, config: RunnableConfig) -> dict[str, object]:
        """在需要确认的 Action 执行前暂停领域子图。"""

        pending_actions = _validated_pending_actions(state)
        owner = _task_owner(config)
        if owner is not None:
            thread_id, user_id = owner
            persisted = await to_thread.run_sync(
                task_state_repository.set_pending_thread_actions,
                thread_id,
                user_id,
                pending_actions,
            )
            if not persisted:
                raise RuntimeError("Unable to persist pending action state")
        decision = interrupt(
            {
                "type": "approval_required",
                "agent_id": agent_id,
                "actions": pending_actions,
                "message": "该操作需要人工确认，请确认后继续。",
            }
        )
        approved = decision is True or (
            isinstance(decision, dict) and decision.get("approved") is True
        )
        reason = decision.get("reason") if isinstance(decision, dict) else None
        if not isinstance(reason, str):
            reason = None
        if owner is not None:
            thread_id, user_id = owner
            resolved = await to_thread.run_sync(
                task_state_repository.resolve_thread_task_approval,
                thread_id,
                user_id,
                approved,
                reason,
            )
            if not resolved:
                raise RuntimeError("Unable to resolve pending action state")
        return {} if approved else {"approval_rejected": True}

    def route_after_model(state: DomainGraphState) -> str:
        """根据模型是否继续调用工具决定下一节点。"""

        # =========================================================================
        # [逻辑规划] 循环结束条件
        # =========================================================================
        # 1. 模型仍有 ToolCall 时，先经过审批检查再执行 ToolNode。
        # 2. 执行过工具且模型不再调用时，说明信息收集完成，进入 Report 汇总全部结果。
        # 3. 首次模型回答没有工具调用时保持普通对话路径，直接结束。
        # =========================================================================
        messages = state.get("execution_messages", [])
        latest = messages[-1] if messages else None
        tool_calls = getattr(latest, "tool_calls", []) or []
        if tool_calls:
            if _validated_pending_actions(state):
                return "approval_gate"
            return "tools"
        if any(isinstance(message, ToolMessage) for message in messages):
            return "report"
        return END

    def route_after_approval(state: DomainGraphState) -> str:
        """根据人工确认结果决定调用工具或交由报告节点结束。"""

        return "report" if state.get("approval_rejected") else "tools"

    graph = StateGraph(DomainGraphState)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    if tools:
        graph.add_node("tools", ToolNode(tools, messages_key="execution_messages"))
        graph.add_node("report", report)
        graph.add_node("approval_gate", approval_gate)
        graph.add_conditional_edges(
            "call_model",
            route_after_model,
            {
                "approval_gate": "approval_gate",
                "tools": "tools",
                "report": "report",
                END: END,
            },
        )
        graph.add_conditional_edges(
            "approval_gate",
            route_after_approval,
            {"tools": "tools", "report": "report"},
        )
        graph.add_edge("tools", "call_model")
        graph.add_edge("report", END)
    else:
        graph.add_edge("call_model", END)
    return graph.compile(checkpointer=checkpointer)


async def _create_orchestrated_agent(
    *,
    model: BaseChatModel,
    settings: Settings,
    checkpointer: Any | None = None,
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
    conversation_graph = await _create_domain_agent(
        agent_id="conversation_agent",
        model=model,
        settings=settings,
    )

    async def call_supervisor(state: AgentGraphState) -> dict[str, object]:
        """调用 Supervisor 并保存结构化路由字段，不把路由结果写入对话消息。"""

        runtime = get_agent_manifest("supervisor").runtime
        try:
            async with asyncio.timeout(runtime.timeout_seconds):
                
                decision = await _invoke_route_decision(
                    supervisor_model,
                    _messages_with_prompt("supervisor", state),
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
        """路由到 IoT 后再发现 MCP 工具并执行领域子图。

        逻辑规划：
        1. Supervisor 已确认目标为 IoT，才建立需要 MCP tools/list 的领域图。
        2. 将父图当前状态和运行配置传入子图，保持多轮工具调用和审批上下文。
        3. 仅回写执行消息；路由决策字段仍由父图持有。
        """

        iot_graph = await _create_domain_agent(
            agent_id="iot_agent",
            model=model,
            settings=settings,
            checkpointer=checkpointer,
        )
        result = await iot_graph.ainvoke(
            {
                "context_messages": state.get("context_messages", []),
                "execution_messages": state.get("execution_messages", []),
            },
            config=config,
        )
        return {"execution_messages": result.get("execution_messages", [])}

    def route_to_agent(state: AgentGraphState) -> str:
        """根据 Supervisor 结果选择唯一的领域 Agent 节点。"""

        target_agent = state.get("target_agent")
        if target_agent not in {"conversation_agent", "iot_agent"}:
            raise ValueError(f"Supervisor returned unsupported target agent: {target_agent}")
        return target_agent

    graph = StateGraph(AgentGraphState)
    graph.add_node("supervisor", call_supervisor)
    graph.add_node("conversation_agent", conversation_graph)
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
    """创建编排图或指定的单领域 Agent 图。

    Args:
        agent_id: 传入时创建指定领域 Agent；不传入时创建 Supervisor 编排图。
        model: 可选的注入模型，便于测试和替换模型实现。
        settings: 可选的已校验配置；未传入时加载默认配置。
        checkpointer: 可选的 LangGraph checkpoint 持久化器。
    Returns:
        已编译的 LangGraph 图。
    """

    resolved_settings = settings or get_settings()
    chat_model = model or _build_model(resolved_settings)
    if agent_id:
        return await _create_domain_agent(
            agent_id=agent_id,
            model=chat_model,
            settings=resolved_settings,
            checkpointer=checkpointer,
        )
    return await _create_orchestrated_agent(
        model=chat_model, settings=resolved_settings, checkpointer=checkpointer
    )
