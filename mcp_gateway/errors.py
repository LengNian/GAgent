"""网关的结构化错误类型：向 Agent 返回可读、可分类的安全错误。

与 app/action_errors.py 保持同构的 error_code / retryable 语义，
但错误文案由网关生成，Agent 侧无需理解中台细节。
"""

from __future__ import annotations


class GatewayError(Exception):
    """网关错误的基类：携带安全文案、稳定编码与是否可重试。

    Args:
        message: 面向模型的安全文案（不含内部细节）。
        error_code: 稳定错误码，供日志与监控聚合。
        retryable: 是否建议模型或上层稍后重试。
    """

    def __init__(self, message: str, *, error_code: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.retryable = retryable


class UpstreamAuthError(GatewayError):
    """与中台的认证交互失败（登录/刷新/凭证无效）。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="upstream_auth_failed", retryable=True)


class ToolAuthError(GatewayError):
    """Agent → 网关的请求方鉴权失败。"""

    def __init__(self, message: str = "网关鉴权失败。") -> None:
        super().__init__(message, error_code="gateway_unauthorized", retryable=False)


class ToolExecutionError(GatewayError):
    """工具执行失败（上游网络、状态码、业务错误码等）。"""


class ToolConfigurationError(GatewayError):
    """网关自身配置错误（不应发生在运行时，出现即说明配置有缺陷）。"""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="gateway_configuration_error", retryable=False)
