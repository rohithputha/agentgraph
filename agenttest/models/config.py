"""
Configuration model for AgentTest.
"""

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class AgentTestConfig:
    """Configuration for AgentTest recording/replay"""

    # Comparison settings
    similarity_threshold: float = 0.85
    default_replay_mode: str = "selective"  # locked, selective, full
    ignore_fields: List[str] = field(default_factory=list)

    # Interceptor configuration
    interceptors: Dict[str, Dict] = field(default_factory=dict)

    # Paths (database is managed by agentgit)
    agentgit_dir: str = ".agentgit"
    project_dir: str = "."

    def to_dict(self) -> dict:
        """Convert config to dictionary for storage"""
        return {
            "similarity_threshold": self.similarity_threshold,
            "default_replay_mode": self.default_replay_mode,
            "ignore_fields": self.ignore_fields,
            "interceptors": self.interceptors,
            "agentgit_dir": self.agentgit_dir,
            "project_dir": self.project_dir
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AgentTestConfig':
        """Create config from dictionary"""
        return cls(
            similarity_threshold=data.get("similarity_threshold", 0.85),
            default_replay_mode=data.get("default_replay_mode", "selective"),
            ignore_fields=data.get("ignore_fields", []),
            interceptors=data.get("interceptors", {}),
            agentgit_dir=data.get("agentgit_dir", ".agentgit"),
            project_dir=data.get("project_dir", ".")
        )