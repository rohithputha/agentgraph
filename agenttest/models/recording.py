"""
Recording model for AgentTest.

A recording represents a test run that captures LLM calls.
Each recording maps to an agentgit branch.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class RecordingStatus(Enum):
    """Status of a recording session"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Recording:
    """Represents a complete recording session"""

    recording_id: str
    name: str
    user_id: str
    session_id: str
    branch_id: int                      # FK to agentgit branches.branch_id
    status: RecordingStatus
    created_at: datetime

    # Optional fields
    completed_at: Optional[datetime] = None
    step_count: int = 0
    error: Optional[str] = None
    config: Optional[dict] = None       # AgentTestConfig snapshot
    metadata: Optional[dict] = None

    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.metadata is None:
            self.metadata = {}
