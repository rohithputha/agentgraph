-- Agent graph schema to store the dag in sqlite database


CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,                
    parent_id INTEGER,                     
    branch_id INTEGER NOT NULL,
    user_id Text NOT NULL,
    session_id Text NOT NULL,
    checkpoint_sha TEXT,
    action_type TEXT NOT NULL,          
    content TEXT NOT NULL,              
    triggered_by TEXT NOT NULL,         
    caller_context TEXT,             
    state_hash TEXT,                    
    timestamp INTEGER NOT NULL,            
    duration_ms INTEGER,                
    token_count INTEGER,                    
    FOREIGN KEY (parent_id) REFERENCES nodes(id)
);


CREATE TABLE IF NOT EXISTS branches (
    branch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    user_id Text NOT NULL,
    session_id Text NOT NULL,
                  
    head_node_id INTEGER,        
    base_node_id INTEGER,
    status TEXT NOT NULL,              
    intent TEXT,                        
    status_reason TEXT,

    created_by TEXT NOT NULL,           
    created_at INTEGER NOT NULL,           
    tokens_used INTEGER DEFAULT 0,
    time_elapsed_seconds REAL DEFAULT 0.0,

    FOREIGN KEY (head_node_id) REFERENCES nodes(id),
    FOREIGN KEY (base_node_id) REFERENCES nodes(id),
    UNIQUE(user_id, session_id, name)
);


CREATE TABLE IF NOT EXISTS checkpoints (
    hash TEXT PRIMARY KEY,              
    node_id INTEGER,                       
    filesystem_ref TEXT,                
    files_changed TEXT,                 
    memory TEXT,                        
    history TEXT,                       
    created_at INTEGER NOT NULL,           
    compressed INTEGER,                 
    size_bytes INTEGER,                 
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

-- ═══════════════════════════════════════════════════════════════
-- Indexes for Performance
-- ═══════════════════════════════════════════════════════════════

-- Query nodes by session
CREATE INDEX IF NOT EXISTS idx_nodes_session
ON nodes(user_id, session_id);

-- Query checkpoints by session
CREATE INDEX IF NOT EXISTS idx_nodes_checkpoint
ON nodes(user_id, session_id, checkpoint_sha) WHERE checkpoint_sha IS NOT NULL;

-- Query nodes by branch
CREATE INDEX IF NOT EXISTS idx_nodes_branch
ON nodes(branch_id);

-- Traverse DAG (get children of node)
CREATE INDEX IF NOT EXISTS idx_nodes_parent
ON nodes(parent_id);

-- Query nodes by time range
CREATE INDEX IF NOT EXISTS idx_nodes_timestamp
ON nodes(timestamp);

-- Query branches by session
CREATE INDEX IF NOT EXISTS idx_branches_session
ON branches(user_id, session_id);

-- Filter branches by status
CREATE INDEX IF NOT EXISTS idx_branches_status
ON branches(status);


-- useful views: review needed

CREATE VIEW IF NOT EXISTS branch_stats AS
SELECT
    b.name,
    b.status,
    b.intent,
    COUNT(n.id) as node_count,
    b.tokens_used,
    b.time_elapsed_seconds,
    b.created_at
FROM branches b
LEFT JOIN nodes n ON n.branch_id = b.branch_id
GROUP BY b.name;

-- View: Recent activity
CREATE VIEW IF NOT EXISTS recent_activity AS
SELECT
    n.id,
    n.action_type,
    n.branch_id,
    b.name as branch_name,
    n.timestamp
FROM nodes n
JOIN branches b ON b.branch_id = n.branch_id
ORDER BY n.timestamp DESC
LIMIT 100;

-- View: Node tree (parent-child relationships)
CREATE VIEW IF NOT EXISTS node_tree AS
SELECT
    n.id,
    n.parent_id,
    n.branch_id,
    n.action_type,
    n.timestamp,
    b.name as branch_name
FROM nodes n
JOIN branches b ON b.branch_id = n.branch_id
ORDER BY n.timestamp;
