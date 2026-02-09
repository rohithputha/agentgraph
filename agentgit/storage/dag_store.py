import sqlite3
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from ..models.dag import (
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

    def insert_node(self, user_id: str, session_id: str, node: ExecutionNode, branch_id: int) -> int:
        """Insert node and return the auto-generated INTEGER id."""
        cursor = self.conn.execute(
            """INSERT INTO nodes (
                user_id, session_id, parent_id, branch_id, checkpoint_sha,
                action_type, content, triggered_by, caller_context, state_hash,
                timestamp, duration_ms, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                session_id,
                int(node.parent_id) if node.parent_id else None,
                branch_id,
                node.checkpoint_sha,
                node.action_type.value,
                json.dumps(node.content),
                node.triggered_by.value,
                json.dumps(node.caller_context),
                node.state_hash,
                int(node.timestamp.timestamp()),
                node.duration_ms,
                node.token_count,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_node(self, user_id: str, session_id: str, node_id: int) -> Optional[ExecutionNode]:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE user_id = ? AND session_id = ? AND id = ?",
            (user_id, session_id, node_id)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def peek(self, user_id: str, session_id: str, node_id: int) -> Optional[dict]:
        """Peek at the memory (content) for a given node number."""
        row = self.conn.execute(
            "SELECT content FROM nodes WHERE user_id = ? AND session_id = ? AND id = ?",
            (user_id, session_id, node_id)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def get_children(self, user_id: str, session_id: str, node_id: int) -> List[ExecutionNode]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE user_id = ? AND session_id = ? AND parent_id = ?",
            (user_id, session_id, node_id)
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def get_branch_nodes(self, user_id: str, session_id: str, branch_id: int) -> List[ExecutionNode]:
        """Get all nodes belonging to a specific branch."""
        rows = self.conn.execute(
            """SELECT * FROM nodes 
               WHERE user_id = ? AND session_id = ? AND branch_id = ? 
               ORDER BY timestamp""",
            (user_id, session_id, branch_id)
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    def get_path_to_root(self, user_id: str, session_id: str, node_id: int) -> List[ExecutionNode]:
        path = []
        current_id: Optional[int] = node_id
        while current_id:
            node = self.get_node(user_id, session_id, current_id)
            if not node:
                break
            path.append(node)
            current_id = int(node.parent_id) if node.parent_id else None
        return list(reversed(path))

    # ─── Branches ─────────────────────────────────────────────────

    def insert_branch(self, user_id: str, session_id: str, branch: Branch) -> int:
        """Insert branch and return the auto-generated branch_id."""
        cursor = self.conn.execute(
            """INSERT INTO branches (
                user_id, session_id, name, head_node_id, base_node_id, status, intent,
                status_reason, created_by, created_at, tokens_used, time_elapsed_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                session_id,
                branch.name,
                int(branch.head_node_id) if branch.head_node_id else None,
                int(branch.base_node_id) if branch.base_node_id else None,
                branch.status.value,
                branch.intent,
                getattr(branch, 'status_reason', None),
                branch.created_by.value,
                int(branch.created_at.timestamp()),
                branch.tokens_used,
                branch.time_elapsed_seconds,
            ),
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_branch(self, user_id: str, session_id: str, name: str) -> Optional[Branch]:
        row = self.conn.execute(
            "SELECT * FROM branches WHERE user_id = ? AND session_id = ? AND name = ?",
            (user_id, session_id, name)
        ).fetchone()
        return self._row_to_branch(row) if row else None

    def get_branch_by_id(self, branch_id: int) -> Optional[Branch]:
        """Get branch by its integer ID."""
        row = self.conn.execute(
            "SELECT * FROM branches WHERE branch_id = ?", (branch_id,)
        ).fetchone()
        return self._row_to_branch(row) if row else None

    def list_branches(self, user_id: str, session_id: str, status: Optional[BranchStatus] = None) -> List[Branch]:
        if status:
            rows = self.conn.execute(
                """SELECT * FROM branches WHERE user_id = ? AND session_id = ? AND status = ? 
                   ORDER BY created_at""",
                (user_id, session_id, status.value),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM branches WHERE user_id = ? AND session_id = ? ORDER BY created_at",
                (user_id, session_id)
            ).fetchall()
        return [self._row_to_branch(r) for r in rows]
    
    def get_active_branch(self, user_id: str, session_id: str) -> Optional[Branch]:
        """Get the active branch for a session (status='active')."""
        row = self.conn.execute(
            """SELECT * FROM branches WHERE user_id = ? AND session_id = ? AND status = ? 
               ORDER BY created_at DESC LIMIT 1""",
            (user_id, session_id, BranchStatus.ACTIVE.value)
        ).fetchone()
        return self._row_to_branch(row) if row else None

    def update_branch_head(self, user_id: str, session_id: str, branch_id: int, new_head_id: int):
        """Update branch head."""
        self.conn.execute(
            """UPDATE branches SET head_node_id = ? 
               WHERE user_id = ? AND session_id = ? AND branch_id = ?""",
            (new_head_id, user_id, session_id, branch_id),
        )
        self.conn.commit()

    def update_branch_status(self, user_id: str, session_id: str, branch_id: int, status: BranchStatus, reason: Optional[str] = None):
        """Update branch status and optional status_reason."""
        self.conn.execute(
            """UPDATE branches SET status = ?, status_reason = ? 
               WHERE user_id = ? AND session_id = ? AND branch_id = ?""",
            (status.value, reason, user_id, session_id, branch_id),
        )
        self.conn.commit()

    # ─── Checkpoints ──────────────────────────────────────────────

    def insert_checkpoint(self, checkpoint: Checkpoint, node_id: int):
        """Insert a checkpoint linked to a node."""
        self.conn.execute(
            """INSERT INTO checkpoints (
                hash, node_id, filesystem_ref, files_changed,
                memory, history, created_at, compressed, size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                checkpoint.hash,
                node_id,
                checkpoint.filesystem_ref,
                json.dumps(checkpoint.files_changed),
                json.dumps(checkpoint.agent_memory),
                json.dumps(checkpoint.conversation_history),
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

    def get_checkpoint_nodes(self, user_id: str, session_id: str) -> List[ExecutionNode]:
        """All CHECKPOINT action type nodes for a session, most recent first."""
        rows = self.conn.execute(
            """SELECT * FROM nodes WHERE user_id = ? AND session_id = ? AND action_type = ? 
               ORDER BY timestamp DESC""",
            (user_id, session_id, "checkpoint"),
        ).fetchall()
        return [self._row_to_node(r) for r in rows]
    
    def get_latest_checkpoint(self, user_id: str, session_id: str) -> Optional[ExecutionNode]:
        """Get most recent checkpoint node for this session (for parent SHA tracking)."""
        row = self.conn.execute(
            """SELECT * FROM nodes WHERE user_id = ? AND session_id = ? AND checkpoint_sha IS NOT NULL 
               ORDER BY timestamp DESC LIMIT 1""",
            (user_id, session_id)
        ).fetchone()
        return self._row_to_node(row) if row else None

    def list_checkpoints(self) -> List[tuple]:
        """List all checkpoints, most recent first."""
        rows = self.conn.execute(
            "SELECT * FROM checkpoints ORDER BY created_at DESC"
        ).fetchall()
        return rows

    # ─── Row → dataclass ──────────────────────────────────────────

    def _row_to_node(self, row) -> ExecutionNode:
        """Map database row to ExecutionNode.
        Schema: id, parent_id, branch_id, user_id, session_id, checkpoint_sha,
                action_type, content, triggered_by, caller_context, state_hash,
                timestamp, duration_ms, token_count
        """
        # Schema indices after our changes:
        # 0:id, 1:parent_id, 2:branch_id, 3:user_id, 4:session_id, 5:checkpoint_sha,
        # 6:action_type, 7:content, 8:triggered_by, 9:caller_context, 10:state_hash,
        # 11:timestamp, 12:duration_ms, 13:token_count

        user_id = row[3]
        session_id = row[4]

        return ExecutionNode(
            user_id=user_id,
            session_id=session_id,
            id=str(row[0]),
            parent_id=str(row[1]) if row[1] else None,
            checkpoint_sha=row[5],
            action_type=ActionType(row[6]),
            content=json.loads(row[7]),
            triggered_by=CallerType(row[8]),
            caller_context=json.loads(row[9]) if row[9] else {},
            state_hash=row[10],
            timestamp=datetime.fromtimestamp(row[11]),
            duration_ms=row[12],
            token_count=row[13],
        )

    def _row_to_branch(self, row) -> Branch:
        """Map database row to Branch.
        Schema: branch_id, name, user_id, session_id, head_node_id, base_node_id,
                status, intent, status_reason, created_by, created_at, tokens_used, time_elapsed_seconds
        """
        # Schema indices:
        # 0:branch_id, 1:name, 2:user_id, 3:session_id, 4:head_node_id, 5:base_node_id,
        # 6:status, 7:intent, 8:status_reason, 9:created_by, 10:created_at, 11:tokens_used, 12:time_elapsed_seconds
        branch = Branch(
            branch_id=row[0],
            user_id=row[2],
            session_id=row[3],
            name=row[1],
            head_node_id=str(row[4]) if row[4] else None,
            base_node_id=str(row[5]) if row[5] else None,
            status=BranchStatus(row[6]),
            intent=row[7] or "",
            created_by=CallerType(row[9]),
            created_at=datetime.fromtimestamp(row[10]),
            tokens_used=row[11] or 0,
            time_elapsed_seconds=row[12] or 0.0,
        )
        # Add status_reason as a dynamic attribute if needed
        if row[8]:  # status_reason
            setattr(branch, 'status_reason', row[8])
        return branch