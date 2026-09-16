"""会话滚动摘要仓储。"""

from uuid import UUID

from app.db.connection import connection as _connection
from app.db.models import ThreadSummary
def load_thread_summary(thread_id: UUID, user_id: str) -> ThreadSummary | None:
    """读取用户所属会话的当前滚动摘要。"""

    with _connection() as connection:
        row = connection.execute(
            """
            SELECT s.summary, s.covered_to_seq, s.summary_version, s.summary_token_count
            FROM aiagent.aiagent_thread_summaries AS s
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = s.thread_id
            WHERE s.thread_id = %s AND t.user_id = %s
            """,
            (thread_id, user_id),
        ).fetchone()
    if row is None:
        return None
    return ThreadSummary(str(row[0]), int(row[1]), int(row[2]), int(row[3]))


def upsert_thread_summary(
    thread_id: UUID,
    user_id: str,
    summary: str,
    covered_to_seq: int,
    summary_token_count: int,
) -> bool:
    """写入更新后的摘要，仅允许覆盖范围向前推进。"""

    with _connection() as connection:
        row = connection.execute(
            """
            INSERT INTO aiagent.aiagent_thread_summaries (
                thread_id, summary, covered_to_seq, summary_version, summary_token_count
            )
            SELECT %s, %s, %s, 1, %s
            WHERE EXISTS (
                SELECT 1 FROM aiagent.aiagent_threads
                WHERE thread_id = %s AND user_id = %s
            )
            ON CONFLICT (thread_id) DO UPDATE
            SET summary = EXCLUDED.summary,
                covered_to_seq = EXCLUDED.covered_to_seq,
                summary_version = aiagent.aiagent_thread_summaries.summary_version + 1,
                summary_token_count = EXCLUDED.summary_token_count,
                updated_at = now()
            WHERE aiagent.aiagent_thread_summaries.covered_to_seq < EXCLUDED.covered_to_seq
            RETURNING thread_id
            """,
            (thread_id, summary, covered_to_seq, summary_token_count, thread_id, user_id),
        ).fetchone()
    return row is not None


