"""
TestStore - Storage layer for AgentTest.

Receives an existing sqlite3.Connection from AgentGit's DagStore.
Creates agenttest-owned tables in the same database.
Reads agentgit tables (nodes, branches) for JOINs but never writes to them.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from agenttest.models.tag import Tag
from agenttest.models.recording import Recording, RecordingStatus
from agenttest.models.llm_call_detail import LLMCallDetail
from agenttest.models.comparison import ComparisonResult, StepComparison, MatchType, StepStatus


class TestStore:
    """
    Storage layer for agenttest tables.
    Shares database connection with AgentGit's DagStore.
    """

    def __init__(self, conn: sqlite3.Connection):
        """
        Initialize TestStore with existing connection.

        Args:
            conn: SQLite connection from DagStore.conn
        """
        self.conn = conn
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._init_schema()

    def _init_schema(self):
        """Load and execute schema.sql for agenttest tables"""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    # ==================== Tags CRUD ====================

    def insert_tag(self, tag: Tag) -> int:
        """Insert a new tag"""
        cursor = self.conn.execute("""
            INSERT INTO at_tags (
                tag_name, user_id, session_id, node_id,
                tag_type, description, metadata, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            tag.tag_name,
            tag.user_id,
            tag.session_id,
            tag.node_id,
            tag.tag_type,
            tag.description,
            json.dumps(tag.metadata) if tag.metadata else None,
            int(tag.created_at.timestamp())
        ))
        self.conn.commit()
        return cursor.lastrowid

    def get_tag(self, user_id: str, session_id: str, tag_name: str) -> Optional[Tag]:
        """Get tag by name"""
        row = self.conn.execute("""
            SELECT * FROM at_tags
            WHERE user_id = ? AND session_id = ? AND tag_name = ?
        """, (user_id, session_id, tag_name)).fetchone()

        return self._row_to_tag(row) if row else None

    def get_tags_for_node(self, node_id: int) -> List[Tag]:
        """Get all tags pointing to a node"""
        rows = self.conn.execute("""
            SELECT * FROM at_tags WHERE node_id = ?
        """, (node_id,)).fetchall()

        return [self._row_to_tag(row) for row in rows]

    def list_tags(
        self,
        user_id: str,
        session_id: str,
        tag_type: Optional[str] = None
    ) -> List[Tag]:
        """List all tags, optionally filtered by type"""
        query = """
            SELECT * FROM at_tags
            WHERE user_id = ? AND session_id = ?
        """
        params = [user_id, session_id]

        if tag_type:
            query += " AND tag_type = ?"
            params.append(tag_type)

        query += " ORDER BY created_at DESC"

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_tag(row) for row in rows]

    def delete_tag(self, user_id: str, session_id: str, tag_name: str) -> bool:
        """Delete a tag"""
        cursor = self.conn.execute("""
            DELETE FROM at_tags
            WHERE user_id = ? AND session_id = ? AND tag_name = ?
        """, (user_id, session_id, tag_name))
        self.conn.commit()
        return cursor.rowcount > 0

    def update_tag(
        self,
        user_id: str,
        session_id: str,
        tag_name: str,
        node_id: int
    ) -> bool:
        """Update tag to point to a different node"""
        cursor = self.conn.execute("""
            UPDATE at_tags
            SET node_id = ?, created_at = ?
            WHERE user_id = ? AND session_id = ? AND tag_name = ?
        """, (node_id, int(datetime.now().timestamp()), user_id, session_id, tag_name))
        self.conn.commit()
        return cursor.rowcount > 0

    # ==================== Recordings CRUD ====================

    def insert_recording(self, recording: Recording) -> str:
        """Insert a new recording"""
        self.conn.execute("""
            INSERT INTO at_recordings (
                recording_id, name, user_id, session_id, branch_id,
                status, created_at, completed_at, step_count,
                error, config, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            recording.recording_id,
            recording.name,
            recording.user_id,
            recording.session_id,
            recording.branch_id,
            recording.status.value,
            int(recording.created_at.timestamp()),
            int(recording.completed_at.timestamp()) if recording.completed_at else None,
            recording.step_count,
            recording.error,
            json.dumps(recording.config) if recording.config else None,
            json.dumps(recording.metadata) if recording.metadata else None
        ))
        self.conn.commit()
        return recording.recording_id

    def get_recording(self, recording_id: str) -> Optional[Recording]:
        """Get recording by ID"""
        row = self.conn.execute("""
            SELECT * FROM at_recordings WHERE recording_id = ?
        """, (recording_id,)).fetchone()

        return self._row_to_recording(row) if row else None

    def get_recording_by_name(
        self,
        user_id: str,
        session_id: str,
        name: str
    ) -> Optional[Recording]:
        """Get recording by name"""
        row = self.conn.execute("""
            SELECT * FROM at_recordings
            WHERE user_id = ? AND session_id = ? AND name = ?
        """, (user_id, session_id, name)).fetchone()

        return self._row_to_recording(row) if row else None

    def list_recordings(
        self,
        user_id: str,
        session_id: str,
        status: Optional[str] = None
    ) -> List[Recording]:
        """List recordings, optionally filtered by status"""
        query = """
            SELECT * FROM at_recordings
            WHERE user_id = ? AND session_id = ?
        """
        params = [user_id, session_id]

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_recording(row) for row in rows]

    def update_recording_status(
        self,
        recording_id: str,
        status: Optional[str] = None,
        error: Optional[str] = None,
        step_count: Optional[int] = None
    ) -> bool:
        """Update recording status and metadata"""
        updates = []
        params = []

        if status:
            updates.append("status = ?")
            params.append(status)
            if status in ["completed", "failed"]:
                updates.append("completed_at = ?")
                params.append(int(datetime.now().timestamp()))

        if error is not None:
            updates.append("error = ?")
            params.append(error)

        if step_count is not None:
            updates.append("step_count = ?")
            params.append(step_count)

        if not updates:
            return False

        query = f"UPDATE at_recordings SET {', '.join(updates)} WHERE recording_id = ?"
        params.append(recording_id)

        cursor = self.conn.execute(query, params)
        self.conn.commit()
        return cursor.rowcount > 0

    def update_recording_step_count(self, recording_id: str, step_count: int) -> bool:
        """
        Update recording step count.

        Note: Does NOT commit. Called during event processing where
        eventbus manages the transaction boundary.
        """
        cursor = self.conn.execute("""
            UPDATE at_recordings SET step_count = ? WHERE recording_id = ?
        """, (step_count, recording_id))
        return cursor.rowcount > 0

    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording (cascades to llm_call_details)"""
        cursor = self.conn.execute("""
            DELETE FROM at_recordings WHERE recording_id = ?
        """, (recording_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    # ==================== LLM Call Details CRUD ====================

    def insert_llm_call_detail(self, detail: LLMCallDetail) -> int:
        """
        Insert LLM call detail.

        Note: Does NOT commit. Called during event processing where
        eventbus manages the transaction boundary.
        """
        cursor = self.conn.execute("""
            INSERT INTO at_llm_call_details (
                node_id, recording_id, step_index, provider, method, model,
                fingerprint, request_params, response_data, is_streaming,
                stream_id, duration_ms, token_usage, error, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detail.node_id,
            detail.recording_id,
            detail.step_index,
            detail.provider,
            detail.method,
            detail.model,
            detail.fingerprint,
            json.dumps(detail.request_params),
            json.dumps(detail.response_data),
            1 if detail.is_streaming else 0,
            detail.stream_id,
            detail.duration_ms,
            json.dumps(detail.token_usage) if detail.token_usage else None,
            detail.error,
            json.dumps(detail.metadata) if detail.metadata else None
        ))
        return cursor.lastrowid

    def get_llm_call_detail(self, detail_id: int) -> Optional[LLMCallDetail]:
        """Get LLM call detail by ID"""
        row = self.conn.execute("""
            SELECT * FROM at_llm_call_details WHERE id = ?
        """, (detail_id,)).fetchone()

        return self._row_to_llm_call_detail(row) if row else None

    def get_llm_call_detail_by_node(self, node_id: int) -> Optional[LLMCallDetail]:
        """Get LLM call detail by node ID"""
        row = self.conn.execute("""
            SELECT * FROM at_llm_call_details WHERE node_id = ?
        """, (node_id,)).fetchone()

        return self._row_to_llm_call_detail(row) if row else None

    def get_recording_details(self, recording_id: str) -> List[LLMCallDetail]:
        """Get all LLM call details for a recording, ordered by step_index"""
        rows = self.conn.execute("""
            SELECT * FROM at_llm_call_details
            WHERE recording_id = ?
            ORDER BY step_index ASC
        """, (recording_id,)).fetchall()

        return [self._row_to_llm_call_detail(row) for row in rows]

    def get_details_by_fingerprint(
        self,
        recording_id: str,
        fingerprint: str
    ) -> List[LLMCallDetail]:
        """Get all LLM call details with matching fingerprint in a recording"""
        rows = self.conn.execute("""
            SELECT * FROM at_llm_call_details
            WHERE recording_id = ? AND fingerprint = ?
            ORDER BY step_index ASC
        """, (recording_id, fingerprint)).fetchall()

        return [self._row_to_llm_call_detail(row) for row in rows]

    # ==================== Comparisons CRUD ====================

    def insert_comparison(
        self,
        result: ComparisonResult,
        user_id: str = "",
        session_id: str = "",
        similarity_threshold: float = 0.85
    ) -> str:
        """Insert comparison result (both main and step comparisons)"""
        # Insert main comparison
        self.conn.execute("""
            INSERT INTO at_comparisons (
                comparison_id, user_id, session_id,
                baseline_recording_id, replay_recording_id,
                overall_pass, similarity_threshold, root_cause_index,
                total_steps, matched_steps, mismatched_steps,
                added_steps, removed_steps, cascade_steps,
                created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.comparison_id,
            user_id,
            session_id,
            result.baseline_recording_id,
            result.replay_recording_id,
            1 if result.overall_pass else 0,
            similarity_threshold,
            result.root_cause_index,
            result.total_steps,
            result.matched_steps,
            result.mismatched_steps,
            result.added_steps,
            result.removed_steps,
            result.cascade_steps,
            int(datetime.now().timestamp()),
            None  # metadata
        ))

        # Insert step comparisons
        for sc in result.step_comparisons:
            self.conn.execute("""
                INSERT INTO at_step_comparisons (
                    comparison_id, step_index,
                    baseline_node_id, replay_node_id,
                    baseline_detail_id, replay_detail_id,
                    status, match_type, similarity_score, diff_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.comparison_id,
                sc.step_index,
                sc.baseline_node_id,
                sc.replay_node_id,
                sc.baseline_detail_id,
                sc.replay_detail_id,
                sc.status.value,
                sc.match_type.value if sc.match_type else None,
                sc.similarity_score,
                sc.diff_summary
            ))

        self.conn.commit()
        return result.comparison_id

    def get_comparison(self, comparison_id: str) -> Optional[ComparisonResult]:
        """Get comparison result by ID"""
        # Get main comparison
        row = self.conn.execute("""
            SELECT * FROM at_comparisons WHERE comparison_id = ?
        """, (comparison_id,)).fetchone()

        if not row:
            return None

        # Get step comparisons
        step_rows = self.conn.execute("""
            SELECT * FROM at_step_comparisons
            WHERE comparison_id = ?
            ORDER BY step_index ASC
        """, (comparison_id,)).fetchall()

        step_comparisons = [self._row_to_step_comparison(r) for r in step_rows]

        return ComparisonResult(
            comparison_id=row['comparison_id'],
            baseline_recording_id=row['baseline_recording_id'],
            replay_recording_id=row['replay_recording_id'],
            overall_pass=bool(row['overall_pass']),
            step_comparisons=step_comparisons,
            root_cause_index=row['root_cause_index'],
            total_steps=row['total_steps'],
            matched_steps=row['matched_steps'],
            mismatched_steps=row['mismatched_steps'],
            added_steps=row['added_steps'],
            removed_steps=row['removed_steps'],
            cascade_steps=row['cascade_steps']
        )

    def list_comparisons(
        self,
        user_id: str,
        session_id: str,
        baseline_recording_id: Optional[str] = None,
        failed_only: bool = False
    ) -> List[ComparisonResult]:
        """List comparisons with optional filters"""
        query = """
            SELECT * FROM at_comparisons
            WHERE user_id = ? AND session_id = ?
        """
        params = [user_id, session_id]

        if baseline_recording_id:
            query += " AND baseline_recording_id = ?"
            params.append(baseline_recording_id)

        if failed_only:
            query += " AND overall_pass = 0"

        query += " ORDER BY created_at DESC"

        rows = self.conn.execute(query, params).fetchall()

        # For list view, don't load step comparisons
        return [
            ComparisonResult(
                comparison_id=row['comparison_id'],
                baseline_recording_id=row['baseline_recording_id'],
                replay_recording_id=row['replay_recording_id'],
                overall_pass=bool(row['overall_pass']),
                step_comparisons=[],  # Empty for list view
                root_cause_index=row['root_cause_index'],
                total_steps=row['total_steps'],
                matched_steps=row['matched_steps'],
                mismatched_steps=row['mismatched_steps'],
                added_steps=row['added_steps'],
                removed_steps=row['removed_steps'],
                cascade_steps=row['cascade_steps']
            )
            for row in rows
        ]

    def get_latest_comparison(
        self,
        user_id: str,
        session_id: str,
        baseline_recording_id: str
    ) -> Optional[ComparisonResult]:
        """Get most recent comparison for a baseline"""
        comparisons = self.list_comparisons(
            user_id, session_id, baseline_recording_id
        )
        return comparisons[0] if comparisons else None

    # ==================== Cross-schema Queries ====================

    def get_recording_with_branch(self, recording_id: str) -> Optional[Dict]:
        """Get recording with branch info (JOINs with agentgit tables)"""
        row = self.conn.execute("""
            SELECT
                r.*,
                b.name as branch_name,
                b.head_node_id,
                b.status as branch_status
            FROM at_recordings r
            JOIN branches b ON r.branch_id = b.branch_id
            WHERE r.recording_id = ?
        """, (recording_id,)).fetchone()

        return dict(row) if row else None

    def get_recording_llm_nodes(self, recording_id: str) -> List[tuple]:
        """Get LLM nodes for a recording (JOINs with agentgit nodes table)"""
        rows = self.conn.execute("""
            SELECT
                n.*,
                d.*
            FROM at_llm_call_details d
            JOIN nodes n ON d.node_id = n.id
            WHERE d.recording_id = ?
            ORDER BY d.step_index ASC
        """, (recording_id,)).fetchall()

        return [dict(row) for row in rows]

    # ==================== Row Converters ====================

    def _row_to_tag(self, row: sqlite3.Row) -> Tag:
        """Convert DB row to Tag"""
        return Tag(
            tag_id=row['tag_id'],
            tag_name=row['tag_name'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            node_id=row['node_id'],
            tag_type=row['tag_type'],
            description=row['description'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            created_at=datetime.fromtimestamp(row['created_at'])
        )

    def _row_to_recording(self, row: sqlite3.Row) -> Recording:
        """Convert DB row to Recording"""
        return Recording(
            recording_id=row['recording_id'],
            name=row['name'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            branch_id=row['branch_id'],
            status=RecordingStatus(row['status']),
            created_at=datetime.fromtimestamp(row['created_at']),
            completed_at=datetime.fromtimestamp(row['completed_at']) if row['completed_at'] else None,
            step_count=row['step_count'],
            error=row['error'],
            config=json.loads(row['config']) if row['config'] else {},
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    def _row_to_llm_call_detail(self, row: sqlite3.Row) -> LLMCallDetail:
        """Convert DB row to LLMCallDetail"""
        return LLMCallDetail(
            id=row['id'],
            node_id=row['node_id'],
            recording_id=row['recording_id'],
            step_index=row['step_index'],
            provider=row['provider'],
            method=row['method'],
            model=row['model'],
            fingerprint=row['fingerprint'],
            request_params=json.loads(row['request_params']),
            response_data=json.loads(row['response_data']),
            is_streaming=bool(row['is_streaming']),
            stream_id=row['stream_id'],
            duration_ms=row['duration_ms'],
            token_usage=json.loads(row['token_usage']) if row['token_usage'] else None,
            error=row['error'],
            metadata=json.loads(row['metadata']) if row['metadata'] else {}
        )

    def _row_to_step_comparison(self, row: sqlite3.Row) -> StepComparison:
        """Convert DB row to StepComparison"""
        return StepComparison(
            step_index=row['step_index'],
            baseline_node_id=row['baseline_node_id'],
            replay_node_id=row['replay_node_id'],
            baseline_detail_id=row['baseline_detail_id'],
            replay_detail_id=row['replay_detail_id'],
            status=StepStatus(row['status']),
            match_type=MatchType(row['match_type']) if row['match_type'] else None,
            similarity_score=row['similarity_score'],
            diff_summary=row['diff_summary']
        )

    def close(self):
        """Close is handled by DagStore since it owns the connection"""
        pass
