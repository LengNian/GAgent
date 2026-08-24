"""请求追踪和结构化日志工具。"""

import json
import logging
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from typing import Any


_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def set_trace_id(trace_id: str) -> Token[str | None]:
    """设置当前异步执行上下文的 trace_id，并返回可用于恢复的 token。"""

    return _trace_id.set(trace_id)


def reset_trace_id(token: Token[str | None]) -> None:
    """恢复进入当前请求前的 trace_id 上下文。"""

    _trace_id.reset(token)


def get_trace_id() -> str | None:
    """返回当前异步执行上下文中的 trace_id。"""

    return _trace_id.get()


class JsonFormatter(logging.Formatter):
    """将标准日志记录编码为可被日志平台解析的 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        """输出时间、级别、模块、trace_id、事件和结构化字段。"""

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": get_trace_id(),
        }
        fields = getattr(record, "structured_fields", None)
        if isinstance(fields, dict):
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """配置应用级 JSON 标准输出日志。"""

    application_logger = logging.getLogger("app")
    if any(isinstance(handler.formatter, JsonFormatter) for handler in application_logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    application_logger.addHandler(handler)
    application_logger.setLevel(logging.INFO)
    application_logger.propagate = False


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: object,
) -> None:
    """写入带结构化字段的日志，字段内容必须是可安全序列化的摘要。"""

    logger.log(level, message, extra={"structured_fields": fields})
