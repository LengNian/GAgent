"""会话审批任务状态仓储。"""

import json
from typing import Any
from uuid import UUID

from app.db.connection import connection as _connection
from app.db.models import ThreadTaskState
def load_thread_task_state(thread_id: UUID, user_id: str) -> ThreadTaskState | None:
    """读取用户所属会话的审批任务状态。"""

    with _connection() as connection:
        row = connection.execute(
            """
            SELECT s.state, s.state_version
            FROM aiagent.aiagent_thread_states AS s
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = s.thread_id
            WHERE s.thread_id = %s AND t.user_id = %s
            """,
            (thread_id, user_id),
        ).fetchone()
    if row is None:
        return None
    state = row[0] if isinstance(row[0], dict) else json.loads(str(row[0]))
    return ThreadTaskState(state, int(row[1]))


def set_pending_thread_actions(
    thread_id: UUID,
    user_id: str,
    pending_actions: list[dict[str, Any]],
) -> bool:
    """保存已校验的待确认 Action，并拒绝覆盖已有待确认状态。"""

    state = {"pending_actions": pending_actions, "approval_status": "pending", "approval_reason": None}
    with _connection() as connection:
        row = connection.execute(
            """
            WITH saved AS (
                INSERT INTO aiagent.aiagent_thread_states (thread_id, state, state_version)
                SELECT %s, %s::jsonb, 1
                WHERE EXISTS (
                    SELECT 1 FROM aiagent.aiagent_threads
                    WHERE thread_id = %s AND user_id = %s
                )
                ON CONFLICT (thread_id) DO UPDATE
                SET state = EXCLUDED.state,
                    state_version = aiagent.aiagent_thread_states.state_version + 1,
                    updated_at = now()
                WHERE aiagent.aiagent_thread_states.state->>'approval_status' <> 'pending'
                RETURNING thread_id
            )
            SELECT thread_id FROM saved
            UNION ALL
            SELECT s.thread_id
            FROM aiagent.aiagent_thread_states AS s
            JOIN aiagent.aiagent_threads AS t ON t.thread_id = s.thread_id
            WHERE s.thread_id = %s
              AND t.user_id = %s
              AND s.state->>'approval_status' = 'pending'
            LIMIT 1
            """,
            (thread_id, json.dumps(state, ensure_ascii=False), thread_id, user_id, thread_id, user_id),
        ).fetchone()
    return row is not None


def resolve_thread_task_approval(
    thread_id: UUID,
    user_id: str,
    approved: bool,
    reason: str | None,
) -> bool:
    """将待审批状态原子推进为批准或拒绝。"""

    status = "approved" if approved else "rejected"
    with _connection() as connection:
        row = connection.execute(
            """
            UPDATE aiagent.aiagent_thread_states
            SET state = jsonb_set(
                    jsonb_set(state, '{approval_status}', to_jsonb(%s::text)),
                    '{approval_reason}', COALESCE(to_jsonb(%s::text), 'null'::jsonb)
                ),
                state_version = state_version + 1,
                updated_at = now()
            WHERE thread_id = %s
              AND (
                  state->>'approval_status' = 'pending'
                  OR (state->>'approval_status' = 'approved' AND %s = TRUE)
              )
              AND EXISTS (
                  SELECT 1 FROM aiagent.aiagent_threads
                  WHERE thread_id = %s AND user_id = %s
              )
            RETURNING thread_id
            """,
            (status, reason, thread_id, approved, thread_id, user_id),
        ).fetchone()
    return row is not None


def finish_thread_task(
    thread_id: UUID,
    user_id: str,
    status: str,
) -> bool:
    """结束批准、拒绝或执行失败后的待确认任务。"""

    if status not in {"completed", "failed", "rejected"}:
        raise ValueError(f"Unsupported task completion status: {status}")
    with _connection() as connection:
        row = connection.execute(
            """
            UPDATE aiagent.aiagent_thread_states
            SET state = jsonb_build_object('approval_status', %s::text),
                state_version = state_version + 1,
                updated_at = now()
            WHERE thread_id = %s
              AND state->>'approval_status' = ANY(%s)
              AND EXISTS (
                  SELECT 1 FROM aiagent.aiagent_threads
                  WHERE thread_id = %s AND user_id = %s
              )
            RETURNING thread_id
            """,
            (status, thread_id, ["approved", "rejected"], thread_id, user_id),
        ).fetchone()
    return row is not None


