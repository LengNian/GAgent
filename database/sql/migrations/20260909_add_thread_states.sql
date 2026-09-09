BEGIN;

CREATE TABLE IF NOT EXISTS aiagent.aiagent_thread_states (
    thread_id     UUID        NOT NULL,
    state         JSONB       NOT NULL,
    state_version INT         NOT NULL DEFAULT 1,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_aiagent_thread_states PRIMARY KEY (thread_id),
    CONSTRAINT fk_aiagent_thread_states_thread FOREIGN KEY (thread_id)
        REFERENCES aiagent.aiagent_threads (thread_id) ON DELETE CASCADE,
    CONSTRAINT chk_aiagent_thread_states_object CHECK (jsonb_typeof(state) = 'object'),
    CONSTRAINT chk_aiagent_thread_states_version CHECK (state_version > 0)
);

DROP TRIGGER IF EXISTS trg_aiagent_thread_states_modify ON aiagent.aiagent_thread_states;
CREATE TRIGGER trg_aiagent_thread_states_modify BEFORE UPDATE ON aiagent.aiagent_thread_states
FOR EACH ROW EXECUTE FUNCTION aiagent.fn_update_modified_at();

COMMIT;
