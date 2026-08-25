"""Action Gateway 授权、参数和执行器边界测试。"""

import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.action_gateway import ActionGateway
from app.action_errors import (
    ActionAuthorizationError,
    ActionError,
    ActionInternalError,
    ActionPreconditionError,
    ActionResponseError,
    ActionTransportError,
    ActionValidationError,
)


class ActionGatewayTests(unittest.IsolatedAsyncioTestCase):
    """验证 Gateway 只执行已授权且参数有效的 Action。"""

    def setUp(self) -> None:
        """创建不包含敏感配置的测试 Settings 替身。"""

        self.settings = MagicMock()

    def test_error_subclasses_have_fixed_error_types(self) -> None:
        """错误类别由异常子类固定，错误码只用于细分具体情形。

        逻辑规划：
        1. 为每个有限错误类别创建一个实例并传入任意具体错误码。
        2. 验证实例均属于 ActionError，确保 Gateway 可统一捕获。
        3. 验证类别来自子类而非调用方传入的可变字符串。
        """

        error_cases = (
            (ActionValidationError, "validation"),
            (ActionPreconditionError, "precondition"),
            (ActionAuthorizationError, "authorization"),
            (ActionTransportError, "transport"),
            (ActionResponseError, "response"),
            (ActionInternalError, "internal"),
        )

        for error_class, expected_type in error_cases:
            error = error_class("测试错误", error_code="test_error")
            self.assertIsInstance(error, ActionError)
            self.assertEqual(error.error_type, expected_type)
            self.assertEqual(error.error_code, "test_error")

    async def test_allowed_action_calls_executor(self) -> None:
        """合法 IPv4 请求应调用 executor 并返回其 JSON 结果。

        逻辑规划：
        1. mock HTTP executor，避免测试依赖真实 NMS 服务。
        2. 通过 iot_agent Gateway 执行已授权 Action。
        3. 确认 executor 被调用一次且结果原样返回。
        """

        request_mock = AsyncMock(return_value={"status": "online"})
        with patch("app.action_gateway.request_tool_json", request_mock):
            gateway = ActionGateway(agent_id="iot_agent", settings=self.settings)
            result = json.loads(
                await gateway.execute(
                    "query_device_by_ip",
                    {"ip": "192.168.1.76"},
                )
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["action_name"], "query_device_by_ip")
        self.assertEqual(result["data"], {"status": "online"})
        request_mock.assert_awaited_once()

    async def test_invalid_ipv4_does_not_call_executor(self) -> None:
        """前置条件失败时不得触发任何外部请求。"""

        request_mock = AsyncMock(return_value={"status": "online"})
        with patch("app.action_gateway.request_tool_json", request_mock):
            gateway = ActionGateway(agent_id="iot_agent", settings=self.settings)
            result = json.loads(
                await gateway.execute(
                    "query_device_by_ip",
                    {"ip": "not-an-ip"},
                )
            )

        self.assertFalse(result["ok"])
        self.assertFalse(result["retryable"])
        self.assertEqual(result["error_type"], "precondition")
        self.assertEqual(result["error_code"], "invalid_ipv4")
        self.assertIn("valid IPv4", result["error"])
        request_mock.assert_not_awaited()

    async def test_transient_executor_error_is_marked_retryable(self) -> None:
        """网络类执行器错误必须标记为可重试，供后续编排使用。"""

        request_mock = AsyncMock(
            side_effect=ActionTransportError(
                "设备查询服务暂时不可用",
                retryable=True,
                error_code="upstream_network_error",
            )
        )
        with patch("app.action_gateway.request_tool_json", request_mock):
            gateway = ActionGateway(agent_id="iot_agent", settings=self.settings)
            result = json.loads(
                await gateway.execute(
                    "query_device_by_ip",
                    {"ip": "192.168.1.76"},
                )
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["retryable"])
        self.assertEqual(result["error_type"], "transport")
        self.assertEqual(result["error_code"], "upstream_network_error")
        self.assertEqual(result["message"], "设备查询服务暂时不可用")

    async def test_unknown_action_is_rejected(self) -> None:
        """未注册 Action 必须被 Gateway 拒绝且不能调用 executor。"""

        request_mock = AsyncMock(return_value={})
        with patch("app.action_gateway.request_tool_json", request_mock):
            gateway = ActionGateway(agent_id="iot_agent", settings=self.settings)
            result = json.loads(await gateway.execute("inject_traffic", {}))

        self.assertFalse(result["ok"])
        self.assertIn("not registered", result["error"])
        request_mock.assert_not_awaited()

    async def test_missing_or_unknown_argument_is_rejected(self) -> None:
        """缺少必填参数或包含额外参数时必须拒绝执行。"""

        request_mock = AsyncMock(return_value={})
        with patch("app.action_gateway.request_tool_json", request_mock):
            gateway = ActionGateway(agent_id="iot_agent", settings=self.settings)
            missing_result = json.loads(
                await gateway.execute("query_device_by_ip", {})
            )
            unknown_result = json.loads(
                await gateway.execute(
                    "query_device_by_ip",
                    {"ip": "192.168.1.76", "extra": "forbidden"},
                )
            )

        self.assertFalse(missing_result["ok"])
        self.assertFalse(unknown_result["ok"])
        request_mock.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
