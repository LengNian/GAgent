ALTER TABLE aiagent.aiagent_messages ADD COLUMN IF NOT EXISTS emotion TEXT;
ALTER TABLE aiagent.aiagent_messages DROP CONSTRAINT IF EXISTS chk_aiagent_messages_emotion;
ALTER TABLE aiagent.aiagent_messages ADD CONSTRAINT chk_aiagent_messages_emotion
CHECK (emotion IS NULL OR emotion IN ('撒娇', '非常高兴', '非常生气', '悲伤', '困惑', '钦佩'));
