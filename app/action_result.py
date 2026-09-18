"""MCP 工具结果的统一结构化契约。"""

from typing import Any, Literal

from pydantic import BaseModel, Field


ActionResultErrorType = Literal[
    "validation",
    "precondition",
    "authorization",
    "transport",
    "response",
    "internal",
]


class ActionResult(BaseModel):
    """MCP 工具返回给 Agent Report 节点的统一结果。"""

    ok: bool
    agent_id: str
    action_name: str
    data: Any | None = None
    error_code: str | None = None
    error_type: ActionResultErrorType | None = None
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    error: str | None = None

    @classmethod
    def success(
        cls,
        *,
        agent_id: str,
        action_name: str,
        data: Any,
    ) -> "ActionResult":
        """构造成功的 MCP 工具结果。

        Args:
            agent_id: 发起调用的 Agent 标识。
            action_name: 已执行的本地工具名。
            data: MCP 网关返回的业务数据。
        Returns:
            可被 Report 节点消费的成功结果。
        """

        # 结果模型只承载网关已返回的数据，不在此处解释或改写业务字段。
        return cls(agent_id=agent_id, action_name=action_name, ok=True, data=data)

    @classmethod
    def failure(
        cls,
        *,
        agent_id: str,
        action_name: str,
        error_code: str,
        error_type: ActionResultErrorType,
        retryable: bool,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> "ActionResult":
        """构造失败的 MCP 工具结果。

        Args:
            agent_id: 发起调用的 Agent 标识。
            action_name: 已执行或尝试执行的本地工具名。
            error_code: 稳定的业务错误码。
            error_type: 失败分类。
            retryable: 是否可安全重试。
            message: 可展示的失败说明。
            details: 非敏感诊断信息。
        Returns:
            可被 Report 节点消费的失败结果。
        """

        # message 同时写入 error，兼容既有 Report 解析的失败结果结构。
        return cls(
            agent_id=agent_id,
            action_name=action_name,
            ok=False,
            error_code=error_code,
            error_type=error_type,
            retryable=retryable,
            details=details or {},
            message=message,
            error=message,
        )

    def to_json(self) -> str:
        """序列化为 ToolMessage 可携带的 JSON 文本。"""

        return self.model_dump_json(ensure_ascii=False)
