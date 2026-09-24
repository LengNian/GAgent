"""Phoenix / OpenTelemetry 分布式追踪接入。

设计边界：
- 本模块只做**追踪基础设施初始化**，不侵入业务代码；未启用时静默跳过。
- 与 `app/observability.py`（结构化日志）正交，两者互补不冲突。
- 通过环境变量控制启用，方便 dev/联调开启、生产默认关闭。
"""

from __future__ import annotations

import os
from pathlib import Path


_tracing_initialized = False

# 项目根目录下的 .env；pydantic-settings 不会将其内容注入 os.environ，
# 而本模块需要在 Settings 之外直接读环境变量，因此显式加载。
_ENV_FILE = Path(__file__).resolve().parent.parent / "config" / ".env"


def _load_env_file() -> None:
    """将 config/.env 加载到 os.environ（不覆盖已有值）。"""

    if not _ENV_FILE.exists():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ENV_FILE, override=False)


def init_tracing() -> None:
    """按环境变量启用 Phoenix trace；未启用时直接返回，零副作用。

    环境变量：
      PHOENIX_ENABLED: "true" 启用；其他值或未设置视为关闭。
      PHOENIX_ENDPOINT: OTLP HTTP 端点，默认 http://localhost:6006/v1/traces。
      PHOENIX_PROJECT_NAME: Phoenix UI 里的 project 分组名，默认 nms-agent。
      PHOENIX_SAMPLE_RATIO: 采样率 0.0~1.0，默认 1.0（全采）。

    幂等：重复调用只生效一次；异常时打印告警但不阻断应用启动。

    逻辑规划：
    1. 全局状态判断是否已初始化，避免重复注册。
    2. 检查开关；关闭时直接返回，不加载任何 OTel 组件。
    3. 懒加载 OTel / OpenInference 依赖，未安装时给出可操作错误提示。
    4. 构建 TracerProvider（Resource + Sampler + BatchSpanProcessor）。
    5. 注册为全局 tracer provider，再通过 LangChainInstrumentor 打补丁。
    """

    global _tracing_initialized
    if _tracing_initialized:
        return
    _load_env_file()
    if os.getenv("PHOENIX_ENABLED", "false").strip().lower() != "true":
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except ImportError as error:
        # 依赖缺失时不阻断应用启动，仅提示运维补齐。
        print(
            f"[tracing] Phoenix 已启用但依赖缺失: {error}. "
            "请安装: pip install openinference-instrumentation-langchain "
            "opentelemetry-sdk opentelemetry-exporter-otlp-proto-http",
            flush=True,
        )
        return

    endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
    project = os.getenv("PHOENIX_PROJECT_NAME", "nms-agent")
    ratio = float(os.getenv("PHOENIX_SAMPLE_RATIO", "1.0"))

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "nms-agent",
                "openinference.project.name": project,
            }
        ),
        sampler=TraceIdRatioBased(ratio),
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=endpoint,
                # Phoenix 通过 x-project-name 头部路由项目；仅靠 Resource 属性在部分版本上不生效。
                headers={"x-project-name": project},
            )
        )
    )
    trace.set_tracer_provider(provider)

    # LangChainInstrumentor 通过 monkey patch 拦截 LangChain / LangGraph
    # 的 Runnable 生命周期，自动产出符合 OpenInference 语义的 span。
    LangChainInstrumentor().instrument()
    _tracing_initialized = True
    print(f"[tracing] Phoenix 已启用 endpoint={endpoint} project={project}", flush=True)


def shutdown_tracing() -> None:
    """关闭时刷新批量 span，避免最后一段 trace 丢失。"""

    if not _tracing_initialized:
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush()
    except Exception:
        # 关闭阶段的任何异常都不应阻断进程退出。
        pass
