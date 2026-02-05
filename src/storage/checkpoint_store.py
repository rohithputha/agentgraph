import json
import gzip
import hashlib
from pathlib import Path
from datetime import datetime

from models.dag import Checkpoint


class CheckpointStore:
    def __init__(self,agit_path: Path ):
        self.agit = agit
    
    def create_checkpoint(self, name: str, agent_memory: Optional[dict] = None):
        """Snapshot agent state. Returns the checkpoint hash."""
        