"""读取并校验 Agent manifest 及其 MCP 工具 allowlist。"""


from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

AGENTS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "agents.yaml"


class AgentRuntimeConfig(BaseModel):
    """Agent 的运行时限制。"""

    max_steps: int = Field(default=5, ge=1, le=20)
    timeout_seconds: float = Field(default=60, gt=0, le=300)


class AgentManifest(BaseModel):
    """描述一个 Agent 身份、职责、Skill 和允许使用的 MCP 工具。"""

    # 小写蛇形id如iot_agent
    agent_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    allowed_actions: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    prompt: str = Field(min_length=1)
    runtime: AgentRuntimeConfig = Field(default_factory=AgentRuntimeConfig)


class AgentsConfig(BaseModel):
    """agents.yaml 的根配置模型。"""

    agents: list[AgentManifest] = Field(min_length=1)

    @model_validator(mode="after")
    def agent_ids_must_be_unique(self) -> "AgentsConfig":
        """拒绝重复 Agent 身份，避免调用方获得不确定的 manifest。"""

        agent_ids = [agent.agent_id for agent in self.agents]
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("agent ids must be unique")
        return self


@lru_cache(maxsize=1)
def get_agents_config() -> AgentsConfig:
    """读取、解析并校验所有 Agent manifest。

    逻辑规划：
    1. 读取 YAML 并校验每个 manifest 的身份、运行限制和 allowlist 类型。
    2. 确认 Agent Prompt 和 Skill 文件都位于受控目录且存在。
    3. 确认 allowlist 中的工具已在 gateways.yaml 声明，避免能力配置不一致。
    4. 返回缓存配置；后续工具构建只从已校验 manifest 获取工具范围。
    """

    if not AGENTS_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Agent manifest configuration not found: {AGENTS_CONFIG_PATH}")

    with AGENTS_CONFIG_PATH.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    # 验证agentid的唯一性
    config = AgentsConfig.model_validate(raw_config)

    from app.prompt_loader import resolve_prompt_path, resolve_skill_path

    for agent in config.agents:
        resolve_prompt_path(agent.prompt)
        for skill_name in agent.skills:
            resolve_skill_path(skill_name)

    # 工具来源已切换为 MCP 网关：allowlist 只需对照网关配置中的工具本地名
    # （网关工具名带平台前缀，如 nms.query_device_by_ip）。发现配置缺失立即
    # 拒绝启动，避免 Agent 声明无法执行的能力。
    from mcp_gateway.config_loader import load_gateway_settings

    gateway_settings = load_gateway_settings(_gateway_config_path())
    gateway_local_names = {
        api.name for platform in gateway_settings.platforms for api in platform.apis
    }

    for agent in config.agents:
        missing = [name for name in agent.allowed_actions if name not in gateway_local_names]
        if missing:
            raise ValueError(
                f"Agent {agent.agent_id} allowlist references tools missing "
                f"from gateway config: {missing}"
            )
    return config


def _gateway_config_path() -> str:
    """返回网关配置文件路径（与网关进程默认路径一致）。"""

    return str(AGENTS_CONFIG_PATH.parent / "gateways.yaml")


@lru_cache(maxsize=None)
def get_agent_manifest(agent_id: str) -> AgentManifest:
    """返回指定 Agent 的 manifest，不存在时抛出明确错误。"""

    for agent in get_agents_config().agents:
        if agent.agent_id == agent_id:
            return agent
    raise KeyError(f"Agent manifest is not registered: {agent_id}")
