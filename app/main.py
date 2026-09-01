"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from pathlib import Path
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.threads import agent_router, router as threads_router
from app.observability import configure_logging
from app.startup import validate_startup_configuration


FRONTEND_DIRECTORY = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """在 FastAPI 启动阶段完成配置校验，再允许服务接收请求。

    逻辑规划：
    1. 调用统一配置校验函数，任何异常都阻止应用进入运行状态。
    2. 配置通过后让 FastAPI 启动并继续提供现有 API。
    3. 当前没有需要在关闭阶段释放的资源，直接结束生命周期。
    """

    configure_logging()
    validate_startup_configuration()
    yield


app = FastAPI(title="General-Agent", lifespan=lifespan)
app.include_router(threads_router)
app.include_router(agent_router)
app.mount("/", StaticFiles(directory=FRONTEND_DIRECTORY, html=True), name="frontend")
