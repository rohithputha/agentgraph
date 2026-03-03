"""PR comment formatting for AgentTest CI results."""

from typing import Dict
from agenttest.models.comparison import ComparisonResult, StepStatus


def format_pr_comment(results: Dict[str, ComparisonResult]) -> str:
    if not results:
        return "## AgentTest\n\nNo test results found."

    any_regression = any(r.has_regression for r in results.values())
    any_delta = any(r.has_delta for r in results.values())
    all_pass = all(r.overall_pass for r in results.values())

    if all_pass:
        return _format_pass(results)

    sections = []
    if any_regression:
        sections.append(_format_regression(results))
    if any_delta:
        sections.append(_format_delta(results))
    return "\n\n---\n\n".join(sections)


def _format_pass(results: Dict[str, ComparisonResult]) -> str:
    total_matched = sum(r.matched_steps for r in results.values())
    total_steps = sum(r.total_steps for r in results.values())
    return (
        "## AgentTest ✅\n\n"
        "All tests passed. No behavior changes detected.\n"
        f"Cached steps: {total_matched}/{total_steps} matched\n"
    )


def _format_regression(results: Dict[str, ComparisonResult]) -> str:
    lines = ["## AgentTest ❌ REGRESSION\n"]
    for name, result in results.items():
        if not result.has_regression:
            continue
        root_idx = result.root_cause_index
        lines.append(f"`{name}` — step {root_idx} diverged from baseline.")
        lines.append("This step was cached and should not have changed.\n")
        if root_idx is not None and root_idx < len(result.step_comparisons):
            step = result.step_comparisons[root_idx]
            lines.append(f"  Node:  step_{root_idx}")
            lines.append(f"  Score: {step.similarity_score:.2f} (threshold: 0.85)")
            if step.diff_summary:
                lines.append(f"  Info:  {step.diff_summary}")
        lines.append("")
        lines.append(
            f"Root cause at step {root_idx}. Steps after may be cascading failures.\n"
        )
    lines.append(
        "**Fix:** investigate why the output changed, then either:\n"
        "- Fix the code so the cached response matches, or\n"
        "- Run `agenttest replay --mode=selective` locally to review and accept\n"
    )
    return "\n".join(lines)


def _format_delta(results: Dict[str, ComparisonResult]) -> str:
    lines = ["## AgentTest ⚠ DELTA — Review Required\n"]
    for name, result in results.items():
        if not result.has_delta:
            continue
        new_steps = [s for s in result.step_comparisons if s.status == StepStatus.ADD]
        live_diverged = [
            s for s in result.step_comparisons
            if s.status == StepStatus.DIVERGE and s.was_cache_hit is False
        ]
        lines.append(f"`{name}` — {len(new_steps)} new step(s), {len(live_diverged)} changed.\n")
        for step in new_steps:
            lines.append(f"  ⚡ Step {step.step_index} (new)")
        for step in live_diverged:
            lines.append(f"  ⚡ Step {step.step_index} (changed, score={step.similarity_score:.2f})")
        cached = result.matched_steps
        total = result.total_steps
        lines.append(f"\nCached steps: {cached}/{total} matched ✓")
        lines.append(f"New/changed steps: {len(new_steps) + len(live_diverged)} (need review)")
        lines.append("")
    lines.append(
        "**To accept:** pull the CI baseline and commit it.\n"
        "```\n"
        "agenttest pull-baseline --from-run <run-id>\n"
        "agenttest accept\n"
        "git add .agentgit/ && git commit\n"
        "```\n"
    )
    return "\n".join(lines)
