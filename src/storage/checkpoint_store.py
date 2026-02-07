import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime
from git_backend import GitBackend
from models.dag import Checkpoint


class CheckpointStore:
    def __init__(self,agit_path: Path, project_dir: Path):
        self.git_back = GitBackend(project_dir, agit_path/ "snapshots.git" )
    
    def create_checkpoint(self, name: str, agent_memory: dict, conversation_history: list, label: str = "Checkpoint"):
        """Snapshot agent state. Returns the checkpoint hash."""
        git_sha = self.git_back.create_snapshot(label)
        checkpoint_data = {
            "git_ref": git_sha,
            "memory": agent_memory,
            "history": conversation_history,
            "label": label,
            "created_at": datetime.now()
        }

        return Checkpoint(**checkpoint_data)
    
    def restore_checkpoint(self, checkpoint: Checkpoint):
        git_sha = checkpoint["git_ref"]        
        if git_sha:
            self.git.restore_snapshot(git_sha)

    