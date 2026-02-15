"""
Comparison models for AgentTest.

Used to represent the results of comparing baseline vs replay recordings.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class MatchType(Enum):
    """How two steps matched"""
    EXACT = "exact"
    SIMILAR = "similar"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class StepStatus(Enum):
    """Status of a step in comparison"""
    MATCH = "match"
    DIVERGE = "diverge"
    ADD = "add"
    REMOVE = "remove"
    CASCADE = "cascade"


@dataclass
class StepComparison:
    """Comparison result for a single step"""

    step_index: int
    baseline_node_id: Optional[int]       # FK to agentgit nodes.id
    replay_node_id: Optional[int]         # FK to agentgit nodes.id
    baseline_detail_id: Optional[int]     # FK to at_llm_call_details.id
    replay_detail_id: Optional[int]       # FK to at_llm_call_details.id
    status: StepStatus
    match_type: Optional[MatchType]
    similarity_score: float

    diff_summary: Optional[str] = None


@dataclass
class ComparisonResult:
    """Result of comparing baseline vs replay recording"""

    comparison_id: str
    baseline_recording_id: str
    replay_recording_id: str

    overall_pass: bool
    step_comparisons: List[StepComparison]

    # Root cause analysis
    root_cause_index: Optional[int] = None

    # Summary statistics
    total_steps: int = 0
    matched_steps: int = 0
    mismatched_steps: int = 0
    added_steps: int = 0
    removed_steps: int = 0
    cascade_steps: int = 0

    def __post_init__(self):
        """Calculate summary stats from step_comparisons if not provided"""
        if self.total_steps == 0:
            self.total_steps = len(self.step_comparisons)
        if self.matched_steps == 0:
            self.matched_steps = sum(1 for s in self.step_comparisons if s.status == StepStatus.MATCH)
        if self.mismatched_steps == 0:
            self.mismatched_steps = sum(1 for s in self.step_comparisons if s.status == StepStatus.DIVERGE)
        if self.added_steps == 0:
            self.added_steps = sum(1 for s in self.step_comparisons if s.status == StepStatus.ADD)
        if self.removed_steps == 0:
            self.removed_steps = sum(1 for s in self.step_comparisons if s.status == StepStatus.REMOVE)
        if self.cascade_steps == 0:
            self.cascade_steps = sum(1 for s in self.step_comparisons if s.status == StepStatus.CASCADE)


