"""网关返回结果前的投影：把中台原始大对象整理成 Agent 可直接引用的结论形状。

投影只改变 data 的内容，不改变工具名、入参 schema 与 ActionResult 契约；
是否投影由 gateways.yaml 中每条 API 的 result_projection 字段声明。
"""

import logging
import math
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 设备身份与型号：只接受字符串文本，数值型 value 一律不认作名称（防止脏值顶替字段）。
DEVICE_NAME_METRICS = ("sysName",)
DEVICE_LOCATION_METRICS = ("sysLocation",)
DEVICE_DESC_METRICS = ("sysDescr",)
DEVICE_MODEL_METRICS = ("hwEntityModelName", "hwEntityBoardName")

# 型号前缀 → 设备类型。依据实测：sysDescr 的 "Versatile Routing Platform" 同时出现在
# 交换机（CloudEngine S5735）与路由器（NetEngine 8000）上，不能用它区分类型。
DEVICE_TYPE_BY_MODEL = (("cloudengine", "switch"), ("netengine", "router"))
DEVICE_TYPE_BY_DESC = (
    ("wireless station", "wireless_station"),
    ("access point", "access_point"),
    ("cloudengine", "switch"),
    ("netengine", "router"),
)

# 无线指标族：用于判定接入方式，不依赖拓扑节点的 icon（实测无线站在拓扑里被标为 server）。
DEVICE_WIRELESS_PREFIXES = ("simWireless", "simMultilink", "dot11")

# ifAdminStatus / ifOperStatus 按 RFC2863：1=up，2=down；其余取值不参与计数，不做语义猜测。
INTERFACE_STATUS_UP = 1
INTERFACE_STATUS_DOWN = 2

DROPPED_NOTE = "已省略指标数值、资源占用与流量类原始序列；如需具体指标请调用 get_realtime_metric 或 get_metric_range。"


def _decode_hex_text(value: Any) -> str | None:
    """还原中台以 0x 十六进制字符串承载的 SNMP octet-string。

    Args:
        value: 待解码文本，可能是 "0x4E6574..." 形式的十六进制串。
    Returns:
        解码后的可读文本；非 hex 形式时原样返回，解码失败时返回 None（不猜测）。
    """
    if not isinstance(value, str) or not value:
        return None
    if not value.startswith("0x"):
        return value
    try:
        decoded = bytes.fromhex(value[2:]).decode("utf-8").strip()
    except ValueError:
        logger.warning("gateway_projection_hex_decode_failed value=%r", value[:40])
        return None
    return decoded or None


def _dedup_by_job(series: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 (metric, labels) 折叠重复序列。

    同一台设备会被多个采集任务（snmp_devices / wired_devices / wireless_devices）重复
    采集，实测会让每条序列原样翻倍；值不一致时保留首次出现并记日志，不静默丢弃。
    """
    seen: dict[Any, dict[str, Any]] = {}
    conflicts = 0
    for item in series:
        if not isinstance(item, dict):
            continue
        labels = item.get("labels")
        key = (item.get("metric"), tuple(sorted(labels.items())) if isinstance(labels, dict) else None)
        first = seen.get(key)
        if first is None:
            seen[key] = item
            continue
        if (first.get("value"), first.get("text")) != (item.get("value"), item.get("text")):
            conflicts += 1
    if conflicts:
        logger.warning("gateway_projection_job_conflict_count=%d", conflicts)
    return list(seen.values())


def _text_of(metrics: dict[str, list[dict[str, Any]]], names: tuple[str, ...]) -> str | None:
    """取指定指标的首个非空字符串值；只有字符串才算文本，数值不参与。"""
    for name in names:
        for item in metrics.get(name, []):
            value = _decode_hex_text(item.get("text"))
            if value:
                return value
    return None


def device_series(data: Any) -> Any:
    """把设备的 SNMP 原始时间序列投影为设备基础信息。

    Args:
        data: 中台 /api/v1/topology/device 返回的 data，形如
            {"common": [...], "wired": [...], "wireless": [...]}。
    Returns:
        含 identity / access / interfaces / neighbors / dropped 的结论字典；
        入参不是字典时原样返回，避免让投影本身成为故障源。
    """
    # =========================================================================
    # [逻辑规划]
    # 1. [防御] 非 dict 直接原样返回：投影不能把可正常回答的请求变成错误。
    # 2. [扁平化] 合并所有列表分组，保留 common→wired→wireless 的出现顺序。
    # 3. [去重] 按 (metric, labels) 折叠多采集任务的重复序列。
    # 4. [身份] sysName / sysLocation / 型号（hex 解码）。
    # 5. [类型] 型号前缀优先，其次 sysDescr 关键词，都判不出才回传 sysDescr 原文。
    # 6. [接入方式] 出现无线指标族判 wireless，否则 wired。
    # 7. [端口] ifNumber 为设备自报总数；up/down 只统计有状态行的端口，
    #    两者不一致时补 note 说明，避免模型把 5/6 当成 1 个 down。
    # 8. [邻居] LLDP 远端系统名去重 + 无线关联 AP 名称，给出数量与清单。
    # 9. [交代] 未纳入的指标种类与序列条数计入 dropped，不静默丢数据。
    # =========================================================================
    if not isinstance(data, dict):
        return data

    grouped = [item for value in data.values() if isinstance(value, list) for item in value]
    series = _dedup_by_job(grouped)
    metrics: dict[str, list[dict[str, Any]]] = {}
    for item in series:
        if isinstance(item.get("metric"), str):
            metrics.setdefault(item["metric"], []).append(item)

    name = _text_of(metrics, DEVICE_NAME_METRICS)
    location = _text_of(metrics, DEVICE_LOCATION_METRICS)
    model = _text_of(metrics, DEVICE_MODEL_METRICS)
    description = _text_of(metrics, DEVICE_DESC_METRICS)

    device_type = None
    for source in (model, description):
        lowered = source.lower() if isinstance(source, str) else ""
        table = DEVICE_TYPE_BY_MODEL if source is model else DEVICE_TYPE_BY_DESC
        for keyword, mapped in table:
            if keyword in lowered:
                device_type = mapped
                break
        if device_type:
            break

    identity: dict[str, Any] = {}
    if name:
        identity["name"] = name
    if device_type:
        identity["type"] = device_type
    elif description:
        # 判不出类型时才回传原文，让模型自己看，而不是留一个空字段诱发猜测。
        identity["description"] = description[:120]
    if model:
        identity["model"] = model
    if location:
        identity["location"] = location

    is_wireless = any(metric.startswith(DEVICE_WIRELESS_PREFIXES) for metric in metrics)

    status_by_index: dict[str, Any] = {}
    for item in metrics.get("ifOperStatus", []):
        labels = item.get("labels")
        if isinstance(labels, dict) and isinstance(labels.get("ifIndex"), str):
            status_by_index.setdefault(labels["ifIndex"], item.get("value"))
    up = sum(1 for value in status_by_index.values() if value == INTERFACE_STATUS_UP)
    down = sum(1 for value in status_by_index.values() if value == INTERFACE_STATUS_DOWN)
    reported = len(status_by_index)
    declared = next((item.get("value") for item in metrics.get("ifNumber", []) if isinstance(item.get("value"), int)), None)
    interfaces: dict[str, Any] = {"total": declared if declared is not None else reported, "up": up, "down": down}
    if declared is not None and reported != declared:
        interfaces["note"] = f"仅 {reported}/{declared} 个端口返回了状态数据，up+down 小于 total 不代表其余端口 down。"

    neighbors = {text for item in metrics.get("lldpRemSysName", []) if isinstance((text := item.get("text")), str) and text}
    neighbors |= {text for item in metrics.get("simWirelessPeerApName", []) if isinstance((text := item.get("text")), str) and text}

    used = {*(DEVICE_NAME_METRICS + DEVICE_LOCATION_METRICS + DEVICE_DESC_METRICS + DEVICE_MODEL_METRICS),
            "ifNumber", "ifOperStatus", "lldpRemSysName", "simWirelessPeerApName"}
    return {
        "identity": identity,
        "access": {"kind": "wireless" if is_wireless else "wired"},
        "interfaces": interfaces,
        "neighbors": {"count": len(neighbors), "names": sorted(neighbors)[:20]},
        "dropped": {
            "metric_kinds": len([metric for metric in metrics if metric not in used]),
            "series_count": max(0, len(series) - sum(len(items) for metric, items in metrics.items() if metric in used)),
            "note": DROPPED_NOTE,
        },
    }


def topology_graph(data: Any) -> dict[str, Any]:
    """提取拓扑节点 IP、名称、状态、类型和链路数量。"""
    if not isinstance(data, dict):
        return {"node_count": 0, "edge_count": 0, "ip_count": 0, "ip_addresses": [], "type_counts": {}, "nodes": []}
    nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else []
    edges = data.get("edges") if isinstance(data.get("edges"), list) else []
    summaries: list[dict[str, str]] = []
    ips: list[str] = []
    statuses: dict[str, int] = {}
    # 类型分布按 icon 在投影侧预先聚合：否则 Report 模型要自己数几十条节点，
    # 实测会把 27 台服务器误数成 24 台。给出权威计数，sum(type_counts) 恒等于 node_count。
    types: dict[str, int] = {}
    for node in nodes:
        node_data = node.get("data") if isinstance(node, dict) else None
        if not isinstance(node_data, dict):
            continue
        summary = {out: node_data[src] for out, src in (("ip", "ip"), ("device_name", "deviceName"), ("label", "label"), ("status", "status"), ("type", "icon")) if isinstance(node_data.get(src), str) and node_data[src]}
        if summary:
            summaries.append(summary)
        if summary.get("ip") and summary["ip"] not in ips:
            ips.append(summary["ip"])
        if summary.get("status"):
            status = summary["status"]
            statuses[status] = statuses.get(status, 0) + 1
        if summary.get("type"):
            device_type = summary["type"]
            types[device_type] = types.get(device_type, 0) + 1
    return {"node_count": len(nodes), "edge_count": len(edges), "status_counts": statuses, "type_counts": types, "ip_count": len(ips), "ip_addresses": ips, "nodes": summaries}


# 指标类工具（get_realtime_metric / get_metric_range）的空结果提示。
# 原因：实测中台对“指标名不存在”“该时段确实未采到数据”一律返回 ok=true + results=[]，
# 模型拿不到区分信号，会把空结果直接说成“设备没有数据”甚至推断成设备异常。
# 这里只陈述事实与表述边界：参数格式类问题（如时间窗口写法）由 argument_guard
# 在发请求前拦住，不在投影里重复——投影文案会被报告模型读到，不应携带参数黑话。
METRIC_EMPTY_RESULTS_NOTE = (
    "本次查询在该时间窗口内没有取到任何数据点。"
    "空结果不能证明设备没有这项数据，也不能据此推断设备异常；"
    "若本轮未核对过该设备的指标名称，先用 list_metric_names 确认这个指标是否存在于这台设备。"
    "向用户表述时应说明“本次未取到数据”，不得说成“设备无数据”或设备故障。"
)


# 单条序列允许送进上下文的最大点数。实测：12 小时窗口按默认 step=15 会有 2881 点/序列，
# 直接撞穿单条消息预算（CONTEXT_MAX_MESSAGE_TOKENS=4096）并被“留头砍尾”，
# 导致模型只能看到 1/12 个序列的开头片段。60 点足够描述形状与趋势。
METRIC_SERIES_POINT_LIMIT = 60

# 全部序列的采样点总预算。只限单序列不够：实测交换机有 6 个真实端口，
# 6×60 点约 9,100 字符；但端口多的设备会到 24 序列×60 点 ≈ 3 万字符，又撞报告预算。
# 按总预算反向收缩每序列采样点，使投影体积不随序列数线性爆炸。
METRIC_TOTAL_SAMPLE_BUDGET = 360

# 展开的最大序列数。实测必要性：48 端口的 range 即使每序列只给 8 点，
# 统计量本身就到 4,710 token，仍超单条消息预算 4,096 会被截断——
# 体积主体是“序列条数”，不是采样点数。
METRIC_SERIES_LIMIT = 12

# 噪声标签：采集任务名、中台自己贴的实例/指标名、以及采集侧角色标签。
# job 尤其关键：实测同一 ifIndex 会被 snmp_devices 与 wired_devices 各采一份，不折叠则序列翻倍；
# 而两个任务贴的标签并不完全相同（wired_devices 多一个 wired_role），所以角色类标签也必须剔出身份，
# 否则折叠失效。失效模式是“退回翻倍现状”，不会把数据算错。
METRIC_NOISE_LABELS = ("job", "instance", "ip", "__name__", "wired_role")

# 时间戳统一转成服务器本地时区的可读格式。原因：报告模型拿到裸 Unix 秒会原样念给用户
# （实测出现过“数据截止时间为 1790213363”）。
METRIC_TIME_FORMAT = "%m-%d %H:%M"


def _metric_number(value: Any) -> float | None:
    """把中台以字符串承载的样本值转成数值；无法转换时返回 None，不猜一个值顶替。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_time_label(seconds: Any) -> str:
    """Unix 秒 → 本地时区可读时间；转换失败回退原值，绝不编造时间。"""

    try:
        return datetime.fromtimestamp(int(float(seconds))).strftime(METRIC_TIME_FORMAT)
    except (TypeError, ValueError, OSError, OverflowError):
        return str(seconds)


def _metric_round(value: float) -> float | int:
    """按量级保留有效位；整数化掉尾零，避免 1.8666666666666667 这类长小数白占 token。"""

    if float(value).is_integer():
        return int(value)
    if abs(value) >= 1000:
        return round(value, 1)
    if abs(value) >= 1:
        return round(value, 2)
    return round(value, 4)


def _metric_fingerprint(points: list[Any]) -> tuple[int, Any, Any]:
    """序列指纹（点数 + 首尾值），用于判断两个采集任务给的是不是同一份数据。"""

    def edge(index: int) -> Any:
        point = points[index]
        return point[1] if isinstance(point, list) and len(point) >= 2 else None

    return (len(points), edge(0), edge(-1) if points else None)


def _metric_fold_job_duplicates(series: list[dict]) -> tuple[list[dict], int]:
    """按去掉噪声标签后的身份折叠重复序列，返回（折叠后列表, 被合并条数）。

    同组保留点数更多的一条；若两组的首尾值或点数不一致，只记 WARNING 不静默丢弃：
    那说明两个采集任务给出的并非同一份数据，合并会掩盖差异。
    """
    grouped: dict[tuple, dict] = {}
    merged = 0
    for item in series:
        raw_labels = item.get("metric") if isinstance(item.get("metric"), dict) else {}
        key = tuple(sorted((k, str(v)) for k, v in raw_labels.items() if k not in METRIC_NOISE_LABELS))
        points = item.get("values") if isinstance(item.get("values"), list) else []
        previous = grouped.get(key)
        if previous is None:
            grouped[key] = item
            continue
        merged += 1
        previous_points = previous.get("values") if isinstance(previous.get("values"), list) else []
        if _metric_fingerprint(previous_points) != _metric_fingerprint(points):
            logger.warning("metric_series 同一指标存在内容不一致的重复序列 labels=%s", key)
        if len(points) > len(previous_points):
            grouped[key] = item
    return list(grouped.values()), merged


def _metric_sample_limit(series_count: int) -> int:
    """按序列数反向算出每序列采样上限，保证总体积有硬上限。

    下限 8 点：再少就连“先升后降”这种形状都描述不了，宁可多占一些预算。
    """

    share = METRIC_TOTAL_SAMPLE_BUDGET // max(series_count, 1)
    return max(8, min(METRIC_SERIES_POINT_LIMIT, share))


def _metric_series_summary(item: dict, sample_limit: int) -> dict[str, Any]:
    """把一条序列压成“身份 + 统计量 + 形状”；统计量始终基于收到的全部原始点计算。

    点数不超上限时 samples 就是全部点（不丢信息，模型传了合适的 step 时不得再抽它）；
    只有超预算才等间隔抽样，这样 min/max/avg 仍反映全貌，尖峰不会因为抽样而被抹掉。
    """
    raw_labels = item.get("metric") if isinstance(item.get("metric"), dict) else {}
    points = [p for p in (item.get("values") or []) if isinstance(p, list) and len(p) >= 2]
    summary: dict[str, Any] = {"labels": {k: v for k, v in raw_labels.items() if k not in METRIC_NOISE_LABELS},
                               "points": len(points)}
    pairs = [(p[0], _metric_number(p[1])) for p in points]
    numbers = [pair for pair in pairs if pair[1] is not None]
    if not numbers:
        return summary
    values = [value for _, value in numbers]
    peak_seconds = max(numbers, key=lambda pair: pair[1])[0]
    summary.update({
        "min": _metric_round(min(values)),
        "max": _metric_round(max(values)),
        "avg": _metric_round(sum(values) / len(values)),
        "peak_at": _metric_time_label(peak_seconds),
        "first": [_metric_time_label(points[0][0]), str(points[0][1])],
        "last": [_metric_time_label(points[-1][0]), str(points[-1][1])],
    })
    if len(values) != len(points):
        summary["invalid_points"] = len(points) - len(values)
    if sample_limit > 0:
        if len(points) <= sample_limit:
            sampled = points
        else:
            stride = math.ceil(len(points) / sample_limit)
            sampled = points[::stride][:sample_limit]
            summary["sampled"] = True
        summary["samples"] = [[_metric_time_label(p[0]), str(p[1])] for p in sampled]
    return summary


def metric_series(data: Any) -> Any:
    """指标查询结果投影：空结果补 note，非空的 range 结果折叠重复序列并聚合趋势。

    逻辑规划：
    1. [形状守卫] 非 dict 或缺少 results 列表时原样返回。
    2. [空结果] results 为空 → 只补 note，保留原始字段，不猜原因。
    3. [形状判定] 序列不是 range 的 {metric, values[]} 形态（realtime 是单点 value）
       时原样返回，保证实时值查询的现有行为零变化。
    4. [折叠+聚合] 先合并同实体的重复采集任务，再逐序列算统计量与抽样。
    5. [交代] 被合并了多少条、是否抽过样，全部写进 note，不让模型误以为是全量数据。

    Args:
        data: 中台返回的 data 字段。
    Returns:
        投影后的结果；不适用投影时原样返回入参。
    """

    if not isinstance(data, dict):
        return data
    results = data.get("results")
    if not isinstance(results, list):
        return data
    if not results:
        return {**data, "note": METRIC_EMPTY_RESULTS_NOTE}
    if not all(isinstance(item, dict) and isinstance(item.get("values"), list) for item in results):
        return data

    folded, merged = _metric_fold_job_duplicates(results)
    # 先只算统计量（不给 samples）以便排序；再只对留下的序列生成采样点，避免白算全量
    profiles = [_metric_series_summary(item, 0) for item in folded]
    # 按“窗口内变化幅度”而非峰值排序：未接线端口的曲线常年是一条直线，
    # 占着预算却什么都说不清；变化幅度最大的序列才是趋势问题需要的信息。
    ranking = sorted(range(len(folded)),
                     key=lambda index: (profiles[index].get("max", 0) - profiles[index].get("min", 0),
                                        profiles[index].get("max", 0)),
                     reverse=True)
    kept = ranking[:METRIC_SERIES_LIMIT]
    sample_limit = _metric_sample_limit(len(kept))
    summaries = [_metric_series_summary(folded[index], sample_limit) for index in kept]
    for summary in summaries:
        summary.pop("sampled", None)
    seconds_start, seconds_end = data.get("start"), data.get("end")
    window: dict[str, Any] = {"step_seconds": data.get("step")}
    if isinstance(seconds_start, (int, float)) and isinstance(seconds_end, (int, float)) and seconds_end >= seconds_start:
        window = {"from": _metric_time_label(seconds_start), "to": _metric_time_label(seconds_end),
                  "hours": _metric_round((seconds_end - seconds_start) / 3600), **window}

    projected: dict[str, Any] = {
        "ip": data.get("ip"),
        "metric": data.get("metric"),
        "rate": data.get("rate"),
        "window": window,
        "series_count": len(summaries),
        "series": summaries,
    }
    notes: list[str] = []
    if merged:
        notes.append(f"已合并 {merged} 条重复采集任务的同实体序列。")
    omitted = len(folded) - len(summaries)
    if omitted > 0:
        notes.append(f"另有 {omitted} 条序列未展开，它们在窗口内的变化幅度均小于已展示的 {len(summaries)} 条。")
    if any(len((folded[index].get("values") or [])) > sample_limit for index in kept):
        notes.append(f"每序列原始点数超过 {sample_limit}，samples 为等间隔抽样；"
                     f"min/max/avg 与 peak_at 基于全部原始点计算。")
    if notes:
        projected["note"] = "".join(notes)
    return projected


PROJECTIONS: dict[str, Callable[[Any], Any]] = {
    "topology_graph": topology_graph,
    "device_series": device_series,
    "metric_series": metric_series,
}
