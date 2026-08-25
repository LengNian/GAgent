"""Action 执行链路的有限错误分类。"""

from typing import Any, ClassVar, Literal


ActionErrorType = Literal[
    "validation",
    "precondition",
    "authorization",
    "transport",
    "response",
    "internal",
]


class ActionError(Exception):
    """Action 可预期失败的基类。

    Args:
        message: 可安全记录和传递给 Report Node 的错误说明。
        error_code: 用于细分错误情形的稳定编码，不对应新的异常类。
        retryable: 当前失败是否适合由编排层再次尝试。
        details: 供日志和报告层使用的非敏感诊断信息。
    """

    error_type: ClassVar[ActionErrorType]

    def __init__(
        self,
        message: str,
        *,
        error_code: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        """保存固定分类和每次失败的细节。"""

        # 逻辑规划：
        # 1. 子类通过类属性固定 error_type，调用处无需传入可变字符串。
        # 2. error_code 保留具体失败原因，避免为每个原因增加异常类。
        # 3. details 仅保存安全诊断字段，后续可由日志与 Report Node 使用。
        self.error_code = error_code
        self.retryable = retryable
        self.details = details or {}
        super().__init__(message)


class ActionValidationError(ActionError):
    """Action 参数的结构、必填字段或类型校验失败。"""

    error_type = "validation"


class ActionPreconditionError(ActionError):
    """Action 的确定性前置条件不满足。"""

    error_type = "precondition"


class ActionAuthorizationError(ActionError):
    """当前 Agent 不具备执行指定 Action 的权限。"""

    error_type = "authorization"


class ActionTransportError(ActionError):
    """外部服务连接、超时或可重试的服务端失败。"""

    error_type = "transport"


class ActionResponseError(ActionError):
    """外部服务返回了不可接受的状态或响应数据。"""

    error_type = "response"


class ActionInternalError(ActionError):
    """Action 注册、执行器配置或本地实现出现内部错误。"""

    error_type = "internal"
