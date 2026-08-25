"""加载共享和 Agent 专属 Prompt。"""

from pathlib import Path

from app.agent_manifest import AgentManifest, get_agent_manifest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIRECTORY = PROJECT_ROOT / "prompts"


def resolve_prompt_path(prompt_reference: str) -> Path:
    """将 manifest 中的 Prompt 相对路径解析到受控 prompts 目录。

    逻辑规划：
    1. 将配置路径解析为绝对路径，统一不同工作目录下的行为。
    2. 确认路径仍位于 prompts 目录内，拒绝 `..` 逃逸读取任意文件。
    3. 确认目标是普通文件，缺失时抛出可定位错误。
    """

    prompt_path = (PROJECT_ROOT / prompt_reference).resolve()
    prompts_root = PROMPTS_DIRECTORY.resolve()
    try:
        prompt_path.relative_to(prompts_root)
    except ValueError as error:
        raise ValueError("Agent prompt must be located under the prompts directory") from error
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Agent prompt not found: {prompt_path}")
    return prompt_path


def load_agent_prompt(manifest: AgentManifest) -> str:
    """加载共享基础 Prompt 和指定 Agent Prompt。"""

    base_prompt_path = resolve_prompt_path("prompts/base.md")
    agent_prompt_path = resolve_prompt_path(manifest.prompt)
    base_prompt = base_prompt_path.read_text(encoding="utf-8").strip()
    agent_prompt = agent_prompt_path.read_text(encoding="utf-8").strip()
    if not base_prompt or not agent_prompt:
        raise ValueError(f"Agent prompt cannot be empty: {manifest.agent_id}")
    return f"{base_prompt}\n\n{agent_prompt}"


def get_agent_prompt(agent_id: str) -> str:
    """按 Agent 身份读取其最终系统 Prompt。"""

    return load_agent_prompt(get_agent_manifest(agent_id))


def get_report_prompt() -> str:
    """加载 Report Node 的受约束摘要 Prompt。"""

    base_prompt = resolve_prompt_path("prompts/base.md").read_text(encoding="utf-8").strip()
    report_prompt = resolve_prompt_path("prompts/report.md").read_text(encoding="utf-8").strip()
    if not base_prompt or not report_prompt:
        raise ValueError("Report prompt cannot be empty")
    return f"{base_prompt}\n\n{report_prompt}"
