"""读取并校验 Agent manifest 及其 Action allowlist。"""
""" 
    快速近似对应：
        涉及到agent, agent_manifest -- agents.yaml
        涉及到ontology -- actions.yaml
        涉及到tools, executor -- tools.yaml
"""


from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from app.ontology import get_action_registry
from app.tools.config import get_tools_config


AGENTS_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "agents.yaml"


class AgentRuntimeConfig(BaseModel):
    """Agent 的运行时限制。"""

    max_steps: int = Field(default=5, ge=1, le=20)
    timeout_seconds: float = Field(default=60, gt=0, le=300)


class AgentManifest(BaseModel):
    """描述一个 Agent 身份、职责和允许使用的 Action。"""

    # 小写蛇形id如iot_agent
    agent_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    allowed_actions: list[str] = Field(default_factory=list)
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
    2. 确认 allowlist 中的 Action 已注册，避免 Agent 声明无法执行的能力。
    3. 确认 Action 引用的底层执行器存在且已启用，阻止配置加载后出现权限幻觉。
    4. 返回缓存配置；后续工具构建只从已校验 manifest 获取 Action 范围。
    """

    if not AGENTS_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Agent manifest configuration not found: {AGENTS_CONFIG_PATH}")

    with AGENTS_CONFIG_PATH.open(encoding="utf-8") as config_file:
        raw_config = yaml.safe_load(config_file)
    # 验证agentid的唯一性
    config = AgentsConfig.model_validate(raw_config)

    # 获得所有可用的action
    action_registry = get_action_registry()
    enabled_executors = {tool.name for tool in get_tools_config().tools if tool.enabled}
    executor_configs = {tool.name: tool for tool in get_tools_config().tools}

    #检查agents的allow_actions与tools里的工具是否一致
    for agent in config.agents:
        for action_name in agent.allowed_actions:
            action = action_registry.get(action_name)
            executor_config = executor_configs.get(action.executor)
            if action.executor not in enabled_executors or executor_config is None:
                raise ValueError(
                    f"Agent {agent.agent_id} Action {action_name} references "
                    f"disabled or missing executor: {action.executor}"
                )
    return config


@lru_cache(maxsize=None)
def get_agent_manifest(agent_id: str) -> AgentManifest:
    """返回指定 Agent 的 manifest，不存在时抛出明确错误。"""

    for agent in get_agents_config().agents:
        if agent.agent_id == agent_id:
            return agent
    raise KeyError(f"Agent manifest is not registered: {agent_id}")
