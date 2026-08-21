"""Agent manifest 和 Action allowlist 测试。"""

import unittest

from app.agent_manifest import get_agent_manifest, get_agents_config
from app.settings import get_settings
from app.tools.registry import build_tools_for_agent


class AgentManifestTests(unittest.TestCase):
    """验证 manifest 能力声明和工具裁剪结果。"""

    def test_iot_manifest_declares_only_device_query(self) -> None:
        """确认 IoT Agent 的 allowlist 只包含当前只读 Action。

        逻辑规划：
        1. 读取已校验的 IoT manifest。
        2. 确认其身份和 allowlist 与配置一致。
        3. 按该身份构建工具，确认模型只获得 allowlist 中的工具。
        """

        manifest = get_agent_manifest("iot_agent")
        tools = build_tools_for_agent(get_settings(), "iot_agent")

        self.assertEqual(manifest.allowed_actions, ["query_device_by_ip"])
        self.assertEqual([tool.name for tool in tools], ["query_device_by_ip"])

    def test_manifest_config_contains_iot_agent(self) -> None:
        """确认 manifest 根配置至少包含一个可用 Agent。"""

        config = get_agents_config()
        self.assertIn("iot_agent", {agent.agent_id for agent in config.agents})

    def test_unknown_agent_is_rejected(self) -> None:
        """确认未注册 Agent 不能获得默认权限。"""

        with self.assertRaises(KeyError):
            get_agent_manifest("unknown_agent")
