CREATE TABLE IF NOT EXISTS user_timed_transcript_cache (
    user_id VARCHAR(36) NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_fingerprint VARCHAR(64) NOT NULL,
    provenance JSONB NOT NULL,
    transcript_data JSONB NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, source_fingerprint, provenance)
);
