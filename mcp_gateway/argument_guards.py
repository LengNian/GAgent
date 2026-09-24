"""网关在请求中台之前对工具入参做的校验：把“中台静默返回空结果”的参数错误变成显式错误。

与 result_projection 对称：由 gateways.yaml 中每条 API 的 argument_guard 字段声明。
校验不通过时抛 ToolExecutionError，该文案会作为工具结果回给 Agent（registry 把网关的
is_error 结果转成 ToolMessage 文本），模型在下一轮据此纠正参数，而不是拿着空数组作答。
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any

from mcp_gateway.config_loader import ApiConfig, PlatformConfig
from mcp_gateway.errors import ToolExecutionError

# 相对时长白名单。实测只有这种写法能被中台按预期解析：start=12h → 窗口跨度正好 12 小时。
RELATIVE_DURATION = re.compile(r"^\d+(\.\d+)?[smhd]$", re.IGNORECASE)

# Unix 秒下界（2001-09-09）。实测填 43200 这类小数值会被判成 HTTP 400，
# 填 -43200 则被当成极早时间，得到跨度 49 万小时的窗口。
MIN_UNIX_SECONDS = 1_000_000_000


def _validate_time_argument(name: str, value: Any) -> str | None:
    """校验单个时间参数，返回错误说明文案；参数合法或未提供时返回 None。

    判定依据全部来自对活网关的实测：
      "12h"     → 窗口跨度 12 小时，有数据
      "-12h"    → 窗口跨度 -12 小时（起点被算到终点之后），0 个数据点且 ok=true
      "12h ago" → 被静默改用默认窗口（最近 1 小时），调用方以为拿到了 12 小时
      "43200"   → HTTP 400
    """

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.startswith("-"):
        positive = text.lstrip("-") or "12h"
        return (
            f"{name}={text!r} 带了负号：中台会把它解析成当前时间之后，"
            f"窗口起点晚于终点，结果是 0 个数据点且不报任何错误。"
            f"想表达“过去 {positive}”请直接写 {positive!r}（不带负号），或用 10 位 Unix 秒填写具体的过去时刻。"
        )
    if RELATIVE_DURATION.match(text):
        return None
    if text.isdigit():
        if int(text) < MIN_UNIX_SECONDS:
            return (
                f"{name}={text!r} 不是有效的 Unix 秒（小于 {MIN_UNIX_SECONDS}）。"
                f"要表达“过去一段时间”，请填 10 位 Unix 秒，或直接用不带负号的相对时长（如 30m、1h、12h、7d）。"
            )
        return None
    return (
        f"{name}={text!r} 无法被识别；中台对无法识别的时间会静默改用默认窗口（最近 1 小时），"
        f"导致你请求的范围和实际拿到的数据不一致。"
        f"只接受不带负号的相对时长（30m、1h、12h、7d）或 10 位 Unix 秒。"
    )


async def guard_time_window(
    *,
    platform: PlatformConfig,
    api: ApiConfig,
    base_url: str,
    token_provider: Any,
    client: Any,
    arguments: dict[str, Any],
) -> None:
    """校验带 start/end 的时间窗口参数（当前用于 get_metric_range）。

    逻辑规划：
    1. [逐参数] start、end 各自按白名单校验，未提供的跳过（中台有默认值）。
    2. [窗口方向] 两者都是 Unix 秒时比较大小，start >= end 属于倒挂，中台会返回空结果。
       - 原因: 相对时长无法在本地换算成绝对时刻（取决于中台的 now），故不做混合形式的比较。
    3. [异常] 全部问题合并成一条错误抛出；不发 HTTP 请求，避免浪费一次注定为空的查询。

    Args:
        platform: 目标中台配置（未使用，保持与所有 guard 同签名）。
        api: 工具对应的 API 映射。
        base_url: 中台 base URL（未使用）。
        token_provider: token 管理器（未使用）。
        client: httpx 客户端（未使用）。
        arguments: 已通过 schema 校验的工具参数。
    Raises:
        ToolExecutionError: 时间参数格式非法或窗口倒挂。
    """

    problems: list[str] = []
    for name in ("start", "end"):
        problem = _validate_time_argument(name, arguments.get(name))
        if problem:
            problems.append(problem)

    start, end = arguments.get("start"), arguments.get("end")
    if not problems and str(start or "").isdigit() and str(end or "").isdigit() and int(start) >= int(end):
        problems.append(f"时间窗口倒挂：start={start} 不早于 end={end}，中台会返回 0 个数据点。")

    if problems:
        raise ToolExecutionError(" ".join(problems), error_code="invalid_time_window", retryable=False)


GUARDS: dict[str, Callable[..., Awaitable[None]]] = {
    "time_window": guard_time_window,
}
