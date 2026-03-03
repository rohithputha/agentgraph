from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List


class MatchType(Enum):
    EXACT = "exact"
    SIMILAR = "similar"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


class StepStatus(Enum):
    MATCH = "match"
    DIVERGE = "diverge"
    ADD = "add"
    REMOVE = "remove"
    CASCADE = "cascade"


@dataclass
class StepComparison:
    step_index: int
    baseline_node_id: Optional[int]
    replay_node_id: Optional[int]
    baseline_detail_id: Optional[int]
    replay_detail_id: Optional[int]
    status: StepStatus
    match_type: Optional[MatchType]
    similarity_score: float
    diff_summary: Optional[str] = None
    was_cache_hit: Optional[bool] = None  # None for REMOVE steps (no replay detail)


@dataclass
class ComparisonResult:
    comparison_id: str
    baseline_recording_id: str
    replay_recording_id: str

    step_comparisons: List[StepComparison] = field(default_factory=list)

    root_cause_index: Optional[int] = None

    total_steps: int = 0
    matched_steps: int = 0
    mismatched_steps: int = 0
    added_steps: int = 0
    removed_steps: int = 0
    cascade_steps: int = 0
    regression_steps: int = 0   # REMOVE + (DIVERGE|CASCADE) where was_cache_hit=True
    delta_steps: int = 0        # ADD + (DIVERGE|CASCADE) where was_cache_hit=False

    created_at: Optional[datetime] = None

    @property
    def has_regression(self) -> bool:
        return self.regression_steps > 0

    @property
    def has_delta(self) -> bool:
        return self.delta_steps > 0

    @property
    def overall_pass(self) -> bool:
        return not self.has_regression and not self.has_delta

    def __post_init__(self):
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
        if self.regression_steps == 0:
            self.regression_steps = sum(
                1 for s in self.step_comparisons
                if s.status == StepStatus.REMOVE
                or (s.status in (StepStatus.DIVERGE, StepStatus.CASCADE) and s.was_cache_hit)
            )
        if self.delta_steps == 0:
            self.delta_steps = sum(
                1 for s in self.step_comparisons
                if s.status == StepStatus.ADD
                or (s.status in (StepStatus.DIVERGE, StepStatus.CASCADE) and s.was_cache_hit is False)
            )
