"""Supervisor 结构化路由。"""
import logging
from typing import Any
from .models import RouteDecision, SupervisorRoutingError
logger = logging.getLogger(__name__)

async def invoke_route_decision(supervisor_model: Any, messages: list[Any]) -> RouteDecision:
    """调用 Supervisor 并校验结构化路由结果，空结果最多重试一次。"""
    for attempt in range(2):
        try:
            decision = await supervisor_model.ainvoke(messages)
        except Exception:
            if attempt == 1:
                raise
            logger.warning("supervisor_invocation_failed_retrying", exc_info=True)
            continue
        if isinstance(decision, RouteDecision):
            return decision
        logger.warning("supervisor_invalid_decision attempt=%s result_type=%s", attempt + 1, type(decision).__name__)
    raise SupervisorRoutingError("Supervisor 未返回有效路由结果")
