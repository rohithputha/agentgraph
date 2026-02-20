"""
LLM Call Detail model for AgentTest.

Stores detailed information about individual LLM calls.
This is a sidecar enrichment for LLM nodes in the DAG.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class LLMCallDetail:
    """Detailed information about an LLM call"""

    id: Optional[int]
    node_id: int                    # FK to agentgit nodes.id
    recording_id: str               # FK to at_recordings.recording_id
    step_index: int                 # Order within recording (0-based)

    # LLM call metadata
    provider: str                   # openai, anthropic, etc.
    method: str                     # chat.completions.create, etc.
    model: str                      # gpt-4, claude-3-opus, etc.

    # Testing-specific data
    fingerprint: str                # Structural hash of request
    request_params: Dict[str, Any]  # Full request parameters
    response_data: Dict[str, Any]   # Full response data

    # Streaming info
    is_streaming: bool = False
    stream_id: Optional[str] = None

    # Performance metrics
    duration_ms: int = 0
    token_usage: Optional[Dict[str, int]] = None

    # Error tracking
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.step_index < 0:
            raise ValueError("step_index must be >= 0")
        if self.metadata is None:
            self.metadata = {}
