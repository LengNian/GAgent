"""应用启动阶段的配置校验。"""

from app.agent_manifest import get_agents_config
from app.ontology import get_ontology_config
from app.settings import get_settings
from app.tools.config import get_tools_config


def validate_startup_configuration() -> None:
    """在服务接受请求前加载并校验全部运行配置。

    逻辑规划：
    1. 加载并校验环境配置，确保模型和外部服务地址可用。
    2. 加载 Ontology/Action 配置，校验对象、Action、schema 和前置条件。
    3. 加载底层 executor 配置，校验 HTTP 方法、地址、超时和重试范围。
    4. 加载 Agent manifest，校验 allowlist、Action 引用和 executor 引用关系。
    5. 任一配置失败都向上抛出异常，阻止应用以半可用状态启动。
    """

    get_settings()
    get_ontology_config()
    get_tools_config()
    get_agents_config()
