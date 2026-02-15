

-- ================================================================
-- TABLE 1: at_tags
-- Git-style refs pointing to nodes. For baselines, milestones, releases.
-- This belongs to agenttest, NOT agentgit.
-- ================================================================
CREATE TABLE IF NOT EXISTS at_tags (
    tag_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    node_id INTEGER NOT NULL,
    tag_type TEXT NOT NULL DEFAULT 'baseline',  -- baseline, release, milestone, custom
    description TEXT,
    metadata TEXT,                               -- JSON
    created_at INTEGER NOT NULL,
    FOREIGN KEY (node_id) REFERENCES nodes(id),
    UNIQUE(user_id, session_id, tag_name)
);

CREATE INDEX IF NOT EXISTS idx_at_tags_session ON at_tags(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_at_tags_type ON at_tags(tag_type);
CREATE INDEX IF NOT EXISTS idx_at_tags_node ON at_tags(node_id);

-- ================================================================
-- TABLE 2: at_recordings
-- Metadata about a test recording. Each recording maps to a branch.
-- A recording is a logical "test run" whose execution nodes live
-- on an agentgit branch.
-- ================================================================
CREATE TABLE IF NOT EXISTS at_recordings (
    recording_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    branch_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'in_progress',  -- in_progress, completed, failed
    created_at INTEGER NOT NULL,
    completed_at INTEGER,
    step_count INTEGER DEFAULT 0,
    error TEXT,
    config TEXT,                                  -- JSON: AgentTestConfig snapshot
    metadata TEXT,                                -- JSON
    FOREIGN KEY (branch_id) REFERENCES branches(branch_id)
);

CREATE INDEX IF NOT EXISTS idx_at_recordings_session
    ON at_recordings(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_at_recordings_branch
    ON at_recordings(branch_id);
CREATE INDEX IF NOT EXISTS idx_at_recordings_name
    ON at_recordings(name);
CREATE INDEX IF NOT EXISTS idx_at_recordings_status
    ON at_recordings(status);

-- ================================================================
-- TABLE 3: at_llm_call_details
-- Sidecar enrichment for LLM nodes. One row per LLM_CALL node.
-- Stores full request/response/fingerprint data needed for testing
-- WITHOUT modifying agentgit's nodes table schema.
-- 1:1 relationship with nodes via UNIQUE(node_id).
-- ================================================================
CREATE TABLE IF NOT EXISTS at_llm_call_details (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id INTEGER NOT NULL UNIQUE,
    recording_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,              -- Order within recording (0-based)
    provider TEXT NOT NULL,                   -- openai, anthropic, etc.
    method TEXT NOT NULL,                     -- chat.completions.create, messages.create
    model TEXT NOT NULL,
    fingerprint TEXT NOT NULL,                -- Structural hash of request
    request_params TEXT NOT NULL,             -- JSON: full request parameters
    response_data TEXT NOT NULL,              -- JSON: full response data
    is_streaming INTEGER NOT NULL DEFAULT 0,
    stream_id TEXT,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    token_usage TEXT,                         -- JSON: {prompt_tokens, completion_tokens, total_tokens}
    error TEXT,
    metadata TEXT,                            -- JSON
    FOREIGN KEY (node_id) REFERENCES nodes(id),
    FOREIGN KEY (recording_id) REFERENCES at_recordings(recording_id)
);

CREATE INDEX IF NOT EXISTS idx_at_llm_details_recording
    ON at_llm_call_details(recording_id);
CREATE INDEX IF NOT EXISTS idx_at_llm_details_fingerprint
    ON at_llm_call_details(fingerprint);
CREATE INDEX IF NOT EXISTS idx_at_llm_details_node
    ON at_llm_call_details(node_id);
CREATE INDEX IF NOT EXISTS idx_at_llm_details_provider
    ON at_llm_call_details(provider);

-- ================================================================
-- TABLE 4: at_comparisons
-- Comparison results between two recordings (baseline vs replay).
-- ================================================================
CREATE TABLE IF NOT EXISTS at_comparisons (
    comparison_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    baseline_recording_id TEXT NOT NULL,
    replay_recording_id TEXT NOT NULL,
    overall_pass INTEGER NOT NULL,            -- 0 or 1
    similarity_threshold REAL NOT NULL,
    root_cause_index INTEGER,                 -- Step index of first divergence
    total_steps INTEGER NOT NULL DEFAULT 0,
    matched_steps INTEGER NOT NULL DEFAULT 0,
    mismatched_steps INTEGER NOT NULL DEFAULT 0,
    added_steps INTEGER NOT NULL DEFAULT 0,
    removed_steps INTEGER NOT NULL DEFAULT 0,
    cascade_steps INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    metadata TEXT,                             -- JSON: test_name, tags, etc.
    FOREIGN KEY (baseline_recording_id) REFERENCES at_recordings(recording_id),
    FOREIGN KEY (replay_recording_id) REFERENCES at_recordings(recording_id)
);

CREATE INDEX IF NOT EXISTS idx_at_comparisons_session
    ON at_comparisons(user_id, session_id);
CREATE INDEX IF NOT EXISTS idx_at_comparisons_baseline
    ON at_comparisons(baseline_recording_id);
CREATE INDEX IF NOT EXISTS idx_at_comparisons_pass
    ON at_comparisons(overall_pass);

-- ================================================================
-- TABLE 5: at_step_comparisons
-- Per-step comparison details within a comparison.
-- References both agentgit nodes and agenttest llm_call_details.
-- ================================================================
CREATE TABLE IF NOT EXISTS at_step_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    comparison_id TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    baseline_node_id INTEGER,                 -- FK nodes.id (nullable for ADDED steps)
    replay_node_id INTEGER,                   -- FK nodes.id (nullable for REMOVED steps)
    baseline_detail_id INTEGER,               -- FK at_llm_call_details.id (nullable)
    replay_detail_id INTEGER,                 -- FK at_llm_call_details.id (nullable)
    status TEXT NOT NULL,                     -- match, diverge, add, remove, cascade
    match_type TEXT,                          -- exact, similar, mismatch, unknown
    similarity_score REAL NOT NULL DEFAULT 0.0,
    diff_summary TEXT,
    FOREIGN KEY (comparison_id) REFERENCES at_comparisons(comparison_id),
    FOREIGN KEY (baseline_node_id) REFERENCES nodes(id),
    FOREIGN KEY (replay_node_id) REFERENCES nodes(id),
    FOREIGN KEY (baseline_detail_id) REFERENCES at_llm_call_details(id),
    FOREIGN KEY (replay_detail_id) REFERENCES at_llm_call_details(id)
);

CREATE INDEX IF NOT EXISTS idx_at_step_comp_comparison
    ON at_step_comparisons(comparison_id);
CREATE INDEX IF NOT EXISTS idx_at_step_comp_status
    ON at_step_comparisons(status);
