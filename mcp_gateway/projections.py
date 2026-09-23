"""网关返回结果前的投影：把中台原始大对象整理成 Agent 可直接引用的结论形状。

投影只改变 data 的内容，不改变工具名、入参 schema 与 ActionResult 契约；
是否投影由 gateways.yaml 中每条 API 的 result_projection 字段声明。
"""

import logging
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
    """提取拓扑节点 IP、名称、状态和链路数量。"""
    if not isinstance(data, dict):
        return {"node_count": 0, "edge_count": 0, "ip_count": 0, "ip_addresses": [], "nodes": []}
    nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else []
    edges = data.get("edges") if isinstance(data.get("edges"), list) else []
    summaries: list[dict[str, str]] = []
    ips: list[str] = []
    statuses: dict[str, int] = {}
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
    return {"node_count": len(nodes), "edge_count": len(edges), "status_counts": statuses, "ip_count": len(ips), "ip_addresses": ips, "nodes": summaries}


PROJECTIONS: dict[str, Callable[[Any], Any]] = {
    "topology_graph": topology_graph,
    "device_series": device_series,
}
