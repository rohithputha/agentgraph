"""
Tag model for AgentTest.

Tags are git-style refs that point to specific nodes in the DAG.
Used for baselines, milestones, releases, and custom markers.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Tag:
    """Git-style tag pointing to a node"""

    tag_id: Optional[int]
    tag_name: str
    user_id: str
    session_id: str
    node_id: int
    tag_type: str  # baseline, release, milestone, custom
    description: Optional[str]
    metadata: Optional[dict]
    created_at: datetime

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
