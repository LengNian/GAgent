"""FastAPI application entry point."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.threads import router as threads_router


FRONTEND_DIRECTORY = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="General-Agent")
app.include_router(threads_router)
app.mount("/", StaticFiles(directory=FRONTEND_DIRECTORY, html=True), name="frontend")
    