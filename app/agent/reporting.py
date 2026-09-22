"""Report 处理接口，作为后续摘要器迁移的稳定边界。"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from app.action_result import ActionResult
from app.prompt_loader import get_report_prompt

ALLOWED_EMOTIONS = frozenset({"撒娇", "非常高兴", "非常生气", "悲伤", "困惑", "钦佩"})

def action_result_from_tool_content(content: object) -> ActionResult | None:
    """解析并校验 ToolMessage 内容。"""
    if isinstance(content, list):
        content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
    try:
        raw = content if isinstance(content, dict) else json.loads(content)
        return ActionResult.model_validate(raw)
    except (TypeError, json.JSONDecodeError, ValidationError):
        return None


def fallback_report_emotion(result: ActionResult | None) -> str:
    """为报告失败场景选择稳定情绪。"""
    if result is None or result.ok:
        return "非常高兴"
    if result.error_type in {"transport", "response"}:
        return "悲伤"
    if result.error_code in {"invalid_ipv4", "missing_required_argument"}:
        return "困惑"
    return "非常生气"


def parse_report_response(content: object, fallback_emotion: str) -> tuple[str, str]:
    """解析 Report 模型的文本或结构化 JSON 响应。"""
    if not isinstance(content, str):
        return "", fallback_emotion
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text, fallback_emotion
    if isinstance(value, dict) and isinstance(value.get("reply"), str):
        emotion = value.get("emotion")
        if emotion not in ALLOWED_EMOTIONS:
            emotion = fallback_emotion
        return value["reply"].strip(), emotion
    return text, fallback_emotion


def report_payload(result: ActionResult, max_chars: int = 81920) -> dict[str, Any]:
    """构造 Report 输入，并对拓扑结果生成专用事实摘要。"""
    data = summarize_topology_graph(result.data) if result.ok and result.action_name == "get_topology_graph" else (result.data if result.ok else None)
    data, truncated = compact_report_value(data, max_chars)
    payload = {"ok": result.ok, "action_name": result.action_name, "data": data,
               "error_code": result.error_code if not result.ok else None,
               "error_type": result.error_type if not result.ok else None,
               "message": result.message if not result.ok else None,
               "details": result.details if not result.ok else {}}
    if truncated:
        payload["data_notice"] = "工具结果过大，报告仅使用首尾代表性数据；如需完整数据请缩小查询范围。"
    if len(json.dumps(payload, ensure_ascii=False)) > max_chars + 1200:
        payload["data"] = "[结果过大，已省略]"
        payload["data_notice"] = "工具结果过大，报告未使用完整数据；请缩小查询范围。"
    return payload


def compact_report_value(value: Any, remaining_chars: int) -> tuple[Any, bool]:
    """在 Report 输入预算内保留结构化数据。"""
    if remaining_chars <= 0:
        return "[结果过大，已省略]", True
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        truncated = False
        for key, item in value.items():
            result[str(key)], item_truncated = compact_report_value(item, remaining_chars)
            truncated = truncated or item_truncated
            if len(json.dumps(result, ensure_ascii=False)) > remaining_chars:
                result.pop(str(key), None)
                result["_truncated"] = "结果过大，部分字段已省略"
                return result, True
        return result, truncated
    if isinstance(value, list):
        if len(json.dumps(value, ensure_ascii=False)) <= remaining_chars:
            return value, False
        return [*value[:10], "[中间数据已省略]", *value[-10:]], True
    if isinstance(value, str) and len(value) > remaining_chars:
        return value[: max(0, remaining_chars - 20)] + "...[已省略]", True
    return value, False


def summarize_topology_graph(data: Any) -> dict[str, Any]:
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


async def report_result_for_messages(messages: list[Any], model: BaseChatModel) -> tuple[str, str]:
    """从工具消息生成最终报告，作为 Report Node 的稳定入口。"""
    results: list[ActionResult] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        content = message.content
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        try:
            raw = content if isinstance(content, dict) else json.loads(content)
            results.append(ActionResult.model_validate(raw))
        except (TypeError, json.JSONDecodeError, ValidationError):
            return "工具返回结果异常，无法确认本次设备查询是否完成。", "困惑"
    if not results:
        return "未收到工具执行结果，无法确认本次设备查询是否完成。", "困惑"
    fallback = next((item for item in results if not item.ok), None)
    fallback_text = "当前操作未能完成，系统已阻止不可信结果返回。" if fallback else "设备查询已完成，但结果摘要生成失败，请稍后重试。"
    fallback_emotion = fallback_report_emotion(fallback)
    try:
        response = await model.ainvoke([
            SystemMessage(content=get_report_prompt()),
            HumanMessage(content=json.dumps({"results": [report_payload(item, max(2000, 81920 // len(results))) for item in results]}, ensure_ascii=False)),
        ])
        text, emotion = parse_report_response(
            getattr(response, "content", ""),
            fallback_emotion,
        )
        if text:
            return text, emotion
    except Exception:
        pass
    return fallback_text, fallback_emotion
