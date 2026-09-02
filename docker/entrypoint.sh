#!/bin/sh
# 容器启动入口：先按需初始化数据库，再启动 FastAPI 应用。
set -e

# 仅在设置了 DATABASE_URL 时初始化（幂等，见 docker/db_init.py）
if [ -n "$DATABASE_URL" ]; then
  python /app/db_init.py || true
fi

# 将 Dockerfile 的 CMD 参数交给 uvicorn 启动
exec "$@"
