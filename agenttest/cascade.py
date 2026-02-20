

from typing import List, Dict, Optional
from agenttest.models.comparison import StepComparison, StepStatus

def detect_root_cause(step_comparisons: List[StepComparison]) -> Optional[int]:
    for i, step_comp in enumerate(step_comparisons):
        if step_comp.status == StepStatus.DIVERGE:
            return i
    return None

def mark_cascades(step_comparisons: List[StepComparison], root_cause_index: Optional[int]) -> None:
    if root_cause_index is None: 
        return

    for i in range(root_cause_index + 1, len(step_comparisons)):
        if step_comparisons[i].status == StepStatus.DIVERGE:
            step_comparisons[i].status = StepStatus.CASCADE
    
def get_cascade_summary(step_comparisons: List[StepComparison]) -> dict:

    root_cause_index = detect_root_cause(step_comparisons)

    total_divergences = sum(
        1 for sc in step_comparisons
        if sc.status in (StepStatus.DIVERGE, StepStatus.CASCADE)
    )

    cascade_count = sum(
        1 for sc in step_comparisons
        if sc.status == StepStatus.CASCADE
    )

    return {
        "root_cause_index": root_cause_index,
        "total_divergences": total_divergences,
        "cascade_count": cascade_count,
        "has_root_cause": root_cause_index is not None
    }