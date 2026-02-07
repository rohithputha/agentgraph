import sqlite3
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from models.dag import (
    ExecutionNode, Branch, ActionType, CallerType, BranchStatus, Checkpoint,
)


class DagStore:
    """Persists execution nodes and branches in SQLite. Loads schema.sql on init."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    # ─── Nodes ────────────────────────────────────────────────────

    def insert_node(self, node: ExecutionNode) -> int:
        """Insert node and return the auto-generated INTEGER id."""
        # Convert node.id from string to int if needed, or let SQLite auto-generate
        # Note: schema uses INTEGER PRIMARY KEY AUTOINCREMENT
        cursor = self.conn.execute(
            """INSERT INTO nodes (
                parent_id, branch_id, action_type, content,
                triggered_by, caller_context, state_hash,
                timestamp, duration_ms, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                int(node.parent_id) if node.parent_id else None,
                # Assuming thread_id maps to branch_id - need to get branch_id from branch name
                self._get_branch_id_from_thread_id(node.thread_id),
                node.action_type.value,
                json.dumps(node.content),
                node.triggered_by.value,
                json.dumps(node.caller_context),
                node.state_hash,
                int(node.timestamp.timestamp()),  # Convert to INTEGER (Unix timestamp)
                node.duration_ms,
                node.token_count,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_node(self, node_id: int) -> Optional[ExecutionNode]:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def peek(self, node_id: int) -> Optional[dict]:
        """Peek at the memory (content) for a given node number.
        
        Returns just the content dict without loading the full ExecutionNode.
        """
        row = self.conn.execute(
            "SELECT content FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_children(self, node_id: int) -> List[ExecutionNode]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE parent_id = ? ORDER BY timestamp",
            (node_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_branch_nodes(self, branch_id: int) -> List[ExecutionNode]:
        """Get all nodes belonging to a specific branch."""
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE branch_id = ? ORDER BY timestamp",
            (branch_id,),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def get_path_to_root(self, node_id: int) -> List[ExecutionNode]:
        path = []
        current_id: Optional[int] = node_id
        while current_id:
            node = self.get_node(current_id)
            if not node:
                break
            path.append(node)
            current_id = int(node.parent_id) if node.parent_id else None
        return list(reversed(path))

    # ─── Branches ─────────────────────────────────────────────────

    def insert_branch(self, branch: Branch) -> int:
        """Insert branch and return the auto-generated branch_id."""
        cursor = self.conn.execute(
            """INSERT INTO branches (
                name, head_node_id, base_node_id,
                status, intent, status_reason, created_by, created_at,
                tokens_used, time_elapsed_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                branch.name,
                int(branch.head_node_id) if branch.head_node_id else None,
                int(branch.base_node_id) if branch.base_node_id else None,
                branch.status.value,
                branch.intent,
                getattr(branch, 'status_reason', None),  # New field in schema
                branch.created_by.value,
                int(branch.created_at.timestamp()),  # Convert to INTEGER (Unix timestamp)
                branch.tokens_used,
                branch.time_elapsed_seconds,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_branch(self, name: str) -> Optional[Branch]:
        row = self.conn.execute(
            "SELECT * FROM branches WHERE name = ?", (name,)
        ).fetchone()
        return self._row_to_branch(row) if row else None

    def get_branch_by_id(self, branch_id: int) -> Optional[Branch]:
        """Get branch by its integer ID."""
        row = self.conn.execute(
            "SELECT * FROM branches WHERE branch_id = ?", (branch_id,)
        ).fetchone()
        return self._row_to_branch(row) if row else None

    def list_branches(self, status: Optional[BranchStatus] = None) -> List[Branch]:
        if status:
            rows = self.conn.execute(
                "SELECT * FROM branches WHERE status = ? ORDER BY created_at",
                (status.value,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM branches ORDER BY created_at"
            ).fetchall()
        return [self._row_to_branch(r) for r in rows]

    def update_branch_head(self, branch_id: int, new_head_id: int):
        """Update branch head using branch_id instead of thread_id."""
        self.conn.execute(
            "UPDATE branches SET head_node_id = ? WHERE branch_id = ?",
            (new_head_id, branch_id),
        )
        self.conn.commit()

    def update_branch_status(self, branch_id: int, status: BranchStatus, reason: Optional[str] = None):
        """Update branch status and optional status_reason."""
        self.conn.execute(
            "UPDATE branches SET status = ?, status_reason = ? WHERE branch_id = ?",
            (status.value, reason, branch_id),
        )
        self.conn.commit()

    # ─── Checkpoints ──────────────────────────────────────────────

    def insert_checkpoint(self, checkpoint: Checkpoint, node_id: int):
        """Insert a checkpoint linked to a node."""
        self.conn.execute(
            """INSERT INTO checkpoints (
                hash, node_id, filesystem_ref, files_changed,
                created_at, compressed, size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.hash,
                node_id,
                checkpoint.filesystem_ref,
                json.dumps(checkpoint.files_changed),
                int(checkpoint.created_at.timestamp()),
                1 if checkpoint.compressed else 0,
                checkpoint.size_bytes,
            ),
        )
        self.conn.commit()

    def get_checkpoint(self, hash: str) -> Optional[tuple]:
        """Get checkpoint by hash. Returns row data."""
        row = self.conn.execute(
            "SELECT * FROM checkpoints WHERE hash = ?", (hash,)
        ).fetchone()
        return row

    def get_checkpoint_nodes(self) -> List[ExecutionNode]:
        """All CHECKPOINT action type nodes, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE action_type = ? ORDER BY timestamp DESC",
            ("checkpoint",),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]

    def list_checkpoints(self) -> List[tuple]:
        """List all checkpoints, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM checkpoints ORDER BY created_at DESC"
        ).fetchall()
        return rows

    # ─── Helper Methods ───────────────────────────────────────────

    def _get_branch_id_from_thread_id(self, thread_id: str) -> int:
        """
        Map thread_id (string) to branch_id (integer).
        This is a transitional helper - ideally the models should be updated to use branch_id directly.
        For now, we'll try to find the branch by name (thread_id) and return its branch_id.
        """
        # Assuming thread_id is the branch name for now
        branch = self.get_branch(thread_id)
        if branch:
            # Get the branch_id from the database
            row = self.conn.execute(
                "SELECT branch_id FROM branches WHERE name = ?", (thread_id,)
            ).fetchone()
            return row[0] if row else 0
        return 0  # Default fallback

    # ─── Row → dataclass ──────────────────────────────────────────

    def _row_to_node(self, row) -> ExecutionNode:
        """Map database row to ExecutionNode. Schema: id, parent_id, branch_id, action_type, content, 
        triggered_by, caller_context, state_hash, timestamp, duration_ms, token_count"""
        # Get the branch to retrieve thread_id (for backward compatibility with models)
        branch = self.get_branch_by_id(row[2])
        thread_id = branch.thread_id if branch else str(row[2])
        
        return ExecutionNode(
            id=str(row[0]),  # Convert INTEGER id to string for compatibility
            parent_id=str(row[1]) if row[1] else None,
            thread_id=thread_id,  # Map branch_id back to thread_id
            action_type=ActionType(row[3]),
            content=json.loads(row[4]),
            triggered_by=CallerType(row[5]),
            caller_context=json.loads(row[6]) if row[6] else {},
            state_hash=row[7],
            timestamp=datetime.fromtimestamp(row[8]),  # Convert INTEGER timestamp to datetime
            duration_ms=row[9],
            token_count=row[10],
        )

    def _row_to_branch(self, row) -> Branch:
        """Map database row to Branch. Schema: branch_id, name, head_node_id, base_node_id, 
        status, intent, status_reason, created_by, created_at, tokens_used, time_elapsed_seconds"""
        branch = Branch(
            name=row[1],
            thread_id=row[1],  # Use name as thread_id for backward compatibility
            head_node_id=str(row[2]) if row[2] else None,  # Convert INTEGER to string
            base_node_id=str(row[3]) if row[3] else None,  # Convert INTEGER to string
            status=BranchStatus(row[4]),
            intent=row[5] or "",
            created_by=CallerType(row[7]),
            created_at=datetime.fromtimestamp(row[8]),  # Convert INTEGER timestamp to datetime
            tokens_used=row[9] or 0,
            time_elapsed_seconds=row[10] or 0.0,
        )
        # Add status_reason as a dynamic attribute if needed
        if row[6]:  # status_reason
            setattr(branch, 'status_reason', row[6])
        return branch