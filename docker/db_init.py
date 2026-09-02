#!/usr/bin/env python3
"""容器启动时按需初始化数据库表结构（combined.sql：关系表 + pgvector 表）。

- 仅当设置了 DATABASE_URL 才执行。
- 幂等：以 combined.sql 最后创建的表 aiagent.knowledge_chunk_embeddings 作为
  "已完整初始化" 的判定标记，存在则跳过，避免容器重启重复建表报错。
  （combined.sql 的 CREATE TABLE 均无 IF NOT EXISTS，故不能用 aiagent.threads 判定，
   否则部分初始化状态下重跑会 already exists 失败。）
- 自动重试若干次，容忍 PG 尚未就绪的启动竞态。
- 初始化失败只告警、不阻塞应用启动（应用首次使用 DB 时才会真正报错）。

前置条件：运维的 PostgreSQL 必须已安装 pgvector 扩展，
否则 combined.sql 的 `CREATE EXTENSION IF NOT EXISTS vector` 会失败。
"""
import os
import sys
import time
from pathlib import Path

SQL_PATH = Path("/app/database/sql/combined.sql")
# combined.sql 中最后创建的表，作为"是否已完整初始化"的判定标记
MARKER_TABLE = "knowledge_chunk_embeddings"
# 关系表：若已存在而标记表不存在，说明之前只跑过 relational.sql（部分初始化）
PARTIAL_TABLE = "threads"
MAX_RETRIES = 5
RETRY_INTERVAL = 2


def _table_exists(cur, table_name: str) -> bool:
    """判断 aiagent schema 下指定表是否已存在。"""

    cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='aiagent' AND table_name=%s",
        (table_name,),
    )
    return cur.fetchone() is not None


def main() -> int:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("[db-init] DATABASE_URL 未设置，跳过数据库初始化。", flush=True)
        return 0
    if not SQL_PATH.exists():
        print(f"[db-init] 找不到 SQL 文件: {SQL_PATH}，跳过初始化。", flush=True)
        return 0

    try:
        from psycopg import connect
    except ImportError:
        print("[db-init] 未安装 psycopg，无法初始化数据库（请确认已安装 psycopg[pool]）。", flush=True)
        return 0

    sql = SQL_PATH.read_text(encoding="utf-8")
    last_err: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with connect(database_url) as conn:
                with conn.cursor() as cur:
                    if _table_exists(cur, MARKER_TABLE):
                        print(
                            f"[db-init] 标记表 aiagent.{MARKER_TABLE} 已存在，判定为已初始化，跳过。",
                            flush=True,
                        )
                        return 0

                    if _table_exists(cur, PARTIAL_TABLE):
                        print(
                            f"[db-init] 警告：aiagent.{PARTIAL_TABLE} 已存在但 aiagent.{MARKER_TABLE} 缺失，"
                            " 数据库处于部分初始化状态，继续建表可能因 already exists 失败；"
                            " 如有报错请先手工补齐或清空后重试。",
                            flush=True,
                        )

                    # 执行 combined.sql：CREATE EXTENSION vector + 全部关系表与 pgvector 表
                    cur.execute(sql)
                conn.commit()
            print("[db-init] 数据库表初始化完成（combined.sql：关系表 + pgvector 表）。", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001 - 启动期容忍并继续
            last_err = exc
            print(f"[db-init] 第 {attempt}/{MAX_RETRIES} 次连接/初始化失败: {exc}", flush=True)
            time.sleep(RETRY_INTERVAL)

    print(f"[db-init] 数据库初始化失败（已重试 {MAX_RETRIES} 次）: {last_err}", flush=True)
    print(
        "[db-init] 应用仍会启动；首次使用数据库时将报错。"
        " 请检查 DATABASE_URL、PG 连通性，以及 PG 是否已安装 pgvector 扩展。",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
