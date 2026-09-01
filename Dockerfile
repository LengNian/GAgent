# NMS Agent —— FastAPI 运行时镜像
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 用 requirements.txt 精确安装，保证镜像与本地 .venv 版本完全一致
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# 复制源码与前端静态资源。
# 注意：config/.env 已在 .dockerignore 中排除，密钥通过运行时环境变量注入，不进镜像层。
COPY app ./app
COPY config ./config
COPY frontend ./frontend
# prompts 为运行时必需（prompt_loader 按 /app/prompts 加载），必须打进镜像
COPY prompts ./prompts

EXPOSE 8000

# 轻量存活探针：根路径由 FastAPI 托管前端，返回 200 即视为存活。
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
