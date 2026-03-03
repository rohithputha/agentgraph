"""GitHub integration for posting PR comments."""

import os
import subprocess
from typing import Optional


def post_pr_comment(comment_body: str, pr_number: Optional[str] = None) -> bool:
    if not pr_number:
        pr_number = os.environ.get("PR_NUMBER") or os.environ.get("GITHUB_PR_NUMBER")
    if not pr_number:
        print("No PR number available. Set PR_NUMBER env var.")
        return False

    result = subprocess.run(
        ["gh", "pr", "comment", str(pr_number), "--body", comment_body],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"gh pr comment failed: {result.stderr}")
        return False
    return True


def get_current_run_id() -> Optional[str]:
    return os.environ.get("GITHUB_RUN_ID")


def get_pr_number() -> Optional[str]:
    return os.environ.get("PR_NUMBER") or os.environ.get("GITHUB_PR_NUMBER")
