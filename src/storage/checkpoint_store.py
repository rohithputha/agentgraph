import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from storage.git_backend import GitBackend
from models.dag import Checkpoint
from storage.dag_store import DagStore


class CheckpointStore:
    def __init__(self, agit_path: Path, project_dir: Path, dag_store: DagStore):
        self.base_dir = agit_path
        self.project_dir = project_dir
        self.dag_store = dag_store
        self.git_back = GitBackend(agit_path / "snapshots.git")
    
    def _get_workspace(self, user_id: str, session_id: str) -> Path:
        """Get isolated workspace for session. Defaults to project_dir if session='default' (optional)."""
        # For true isolation, always use separate dirs.
        # But for CLI convenience (editing local files), 'default' could map to project_dir.
        # Let's enforce isolation for now to be safe.
        params = [p for p in (user_id, session_id) if p and p != "default"]
        if not params:
            # If default/default, maybe use project_dir? 
            # Let's stick to using project_dir for default session for backward compatibility/CLI usage.
            return self.project_dir

        ws = self.base_dir / "workspaces" / user_id / session_id
        ws.mkdir(parents=True, exist_ok=True)
        return ws
    
    def create_checkpoint(
        self,
        user_id: str,
        session_id: str,
        name: str,
        agent_memory: dict,
        conversation_history: list,
        label: str = "Checkpoint"
    ) -> Checkpoint:
        """Snapshot agent state. Returns the Checkpoint object."""
        workspace = self._get_workspace(user_id, session_id)
        
        # 1. Get parent SHA from DagStore
        parent_node = self.dag_store.get_latest_checkpoint(user_id, session_id)
        parent_sha = parent_node.checkpoint_sha if parent_node else None
        
        # 2. Create commit (stateless)
        git_sha = self.git_back.create_commit(workspace, parent_sha, label)
        
        # Create hash from memory + history
        content_str = json.dumps({"memory": agent_memory, "history": conversation_history}, sort_keys=True)
        checkpoint_hash = hashlib.sha256(content_str.encode()).hexdigest()[:12]
        
        checkpoint = Checkpoint(
            hash=checkpoint_hash,
            agent_memory=agent_memory,
            conversation_history=conversation_history,
            filesystem_ref=git_sha,
            files_changed=[],  # Could be populated from git diff
            created_at=datetime.now(),
            compressed=False,
            size_bytes=len(content_str),
            label=label,
        )
        
        return checkpoint
    
    def restore_checkpoint(self, checkpoint: Checkpoint, user_id: str = "default", session_id: str = "default"):
        """Restore filesystem state from a checkpoint."""
        workspace = self._get_workspace(user_id, session_id)
        if checkpoint.filesystem_ref:
            self.git_back.restore_commit(checkpoint.filesystem_ref, workspace)