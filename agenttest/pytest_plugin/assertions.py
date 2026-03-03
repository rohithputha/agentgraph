from typing import Optional

from agenttest.models.comparison import ComparisonResult, MatchType


def assert_no_regression(
    comparison_result: ComparisonResult,
    message: Optional[str] = None
) -> None:
    if not comparison_result.overall_pass:
        error_msg = message or f"Regression detected: {_get_summary(comparison_result)}"
        raise AssertionError(error_msg)


def assert_step_count(
    comparison_result: ComparisonResult,
    min_steps: Optional[int] = None,
    max_steps: Optional[int] = None,
    exact_steps: Optional[int] = None
) -> None:
    total = comparison_result.total_steps

    if exact_steps is not None:
        if total != exact_steps:
            raise AssertionError(
                f"Expected exactly {exact_steps} steps, got {total}"
            )

    if min_steps is not None and total < min_steps:
        raise AssertionError(
            f"Expected at least {min_steps} steps, got {total}"
        )

    if max_steps is not None and total > max_steps:
        raise AssertionError(
            f"Expected at most {max_steps} steps, got {total}"
        )


def assert_no_new_errors(
    comparison_result: ComparisonResult
) -> None:
    for step in comparison_result.step_comparisons:
        if step.replay_error and not step.baseline_error:
            raise AssertionError(
                f"New replay error at step {step.step_index}: {step.replay_error}"
            )


def assert_exact_match(
    comparison_result: ComparisonResult
) -> None:
    for step in comparison_result.step_comparisons:
        if step.match_type != MatchType.EXACT:
            raise AssertionError(
                f"Step {step.step_index} is not an exact match: {step.diff_summary}"
            )


def assert_similarity_above(
    comparison_result: ComparisonResult,
    threshold: float
) -> None:
    for step in comparison_result.step_comparisons:
        if step.similarity_score < threshold:
            raise AssertionError(
                f"Step {step.step_index} similarity {step.similarity_score:.2f} "
                f"below threshold {threshold:.2f}"
            )


def _get_summary(comparison_result: ComparisonResult) -> str:
    if comparison_result.root_cause_index is not None:
        root_step = comparison_result.step_comparisons[comparison_result.root_cause_index]
        return f"Step {root_step.step_index}: {root_step.diff_summary}"
    return f"{comparison_result.mismatched_steps} divergences found"
