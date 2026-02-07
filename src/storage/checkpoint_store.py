import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional
from storage.git_backend import GitBackend
from models.dag import Checkpoint


class CheckpointStore:
    def __init__(self, agit_path: Path, project_dir: Path):
        self.git_back = GitBackend(project_dir, agit_path / "snapshots.git")
    
    def create_checkpoint(
        self,
        name: str,
        agent_memory: dict,
        conversation_history: list,
        label: str = "Checkpoint"
    ) -> Checkpoint:
        """Snapshot agent state. Returns the Checkpoint object."""
        git_sha = self.git_back.create_snapshot(label)
        
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
    
    def restore_checkpoint(self, checkpoint: Checkpoint):
        """Restore filesystem state from a checkpoint."""
        if checkpoint.filesystem_ref:
            self.git_back.restore_snapshot(checkpoint.filesystem_ref)