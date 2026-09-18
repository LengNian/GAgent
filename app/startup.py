"""应用启动阶段的配置校验。"""

from app.agent_manifest import get_agents_config
from app.settings import get_settings


def validate_startup_configuration() -> None:
    """在服务接受请求前加载并校验全部运行配置。

    逻辑规划：
    1. 加载并校验环境配置，确保模型和外部服务地址可用。
    2. 加载 Agent manifest；其会校验 allowlist 与 gateways.yaml 的 MCP 工具契约。
    3. 任一配置失败都向上抛出异常，阻止应用以半可用状态启动。
    """

    get_settings()
    get_agents_config()
