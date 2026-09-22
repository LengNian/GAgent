"""领域 Agent 子图的构造与运行边界。"""

import asyncio
from langgraph.errors import GraphRecursionError
from typing import Any
from uuid import UUID

from anyio import to_thread
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import ValidationError

from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.agent.models import AgentExecutionLimitError, DomainGraphState
from app.agent.message_builder import messages_with_prompt
from app.agent.reporting import action_result_from_tool_content, report_result_for_messages
from app.agent_manifest import get_agent_manifest
from app.db.repositories import task_state_repository
from app.settings import Settings
from app.tools.registry import build_tools_for_agent


async def invoke_domain_graph(
    *,
    agent_id: str,
    graph: Any,
    messages: list[Any],
    config: RunnableConfig | None = None,
    manifest_getter: Any | None = None,
) -> dict[str, Any]:
    """按 Agent manifest 的超时和最大步骤限制执行领域图。"""

    runtime = (manifest_getter or get_agent_manifest)(agent_id).runtime
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
            agent_id, "timeout_seconds", runtime.timeout_seconds
        ) from error
    except GraphRecursionError as error:
        raise AgentExecutionLimitError(
            agent_id, "max_steps", runtime.max_steps
        ) from error


async def create_domain_graph(
    *,
    agent_id: str,
    model: BaseChatModel,
    settings: Settings,
    checkpointer: Any | None = None,
    tool_builder: Any | None = None,
):
    """创建单个领域 Agent 图。

    领域图负责模型规划、MCP 工具执行、审批中断以及最终 Report 汇总；
    Supervisor 只负责选择领域，不参与这些执行细节。
    """

    resolved_tool_builder = tool_builder or build_tools_for_agent
    tools = await resolved_tool_builder(settings, agent_id)
    model_with_tools = model.bind_tools(tools) if tools else model
    tool_policies = {tool.name: tool.metadata or {} for tool in tools}
    tools_by_name = {tool.name: tool for tool in tools}

    def _task_owner(config: RunnableConfig) -> tuple[UUID, str] | None:
        """从运行配置读取审批状态归属；配置缺失时不持久化任务状态。"""

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
        """校验模型提出的工具调用，并筛选需要人工确认的调用。"""

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
        """在领域图结束时清除审批任务状态。"""

        owner = _task_owner(config)
        if owner is None:
            return
        thread_id, user_id = owner
        await to_thread.run_sync(task_state_repository.finish_thread_task, thread_id, user_id, status)

    async def call_model(state: DomainGraphState) -> dict[str, list[Any]]:
        """调用领域模型，决定继续调用工具还是结束信息收集。"""

        has_tool_result = any(
            isinstance(message, ToolMessage)
            for message in state.get("execution_messages", [])
        )
        config = {"tags": ["tool_planning"]} if has_tool_result else None
        response = await model_with_tools.ainvoke(
            messages_with_prompt(agent_id, state),
            config=config,
        )
        return {"execution_messages": [response]}

    async def report(state: DomainGraphState, config: RunnableConfig) -> dict[str, list[Any]]:
        """根据实际 ToolMessage 生成最终用户报告。"""

        if state.get("approval_rejected"):
            await _finish_task(config, "rejected")
            return {
                "execution_messages": [
                    AIMessage(
                        content="操作已取消，未执行相关工具。",
                        additional_kwargs={"emotion": "悲伤"},
                    )
                ]
            }
        report_message, emotion = await report_result_for_messages(
            state.get("execution_messages", []), model
        )
        action_results = [
            action_result_from_tool_content(message.content)
            for message in state.get("execution_messages", [])
            if isinstance(message, ToolMessage)
        ]
        completed = bool(action_results) and all(
            result is not None and result.ok for result in action_results
        )
        await _finish_task(config, "completed" if completed else "failed")
        return {
            "execution_messages": [
                AIMessage(
                    content=report_message,
                    additional_kwargs={"emotion": emotion},
                )
            ]
        }

    async def approval_gate(state: DomainGraphState, config: RunnableConfig) -> dict[str, object]:
        """持久化待确认调用并暂停图，直到收到人工审批结果。"""

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
        """根据模型输出选择审批、工具、报告或直接结束。"""

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
        """审批拒绝进入报告，批准后执行工具。"""

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
