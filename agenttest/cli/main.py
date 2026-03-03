"""
AgentTest CLI - Main entry point
"""
import sys
import click
from datetime import datetime
from typing import Optional

from agenttest.session import AgentTestSession


def _make_session(project_dir: str) -> AgentTestSession:
    return AgentTestSession.standalone(project_dir=project_dir)


def _fmt_id(id_str: Optional[str], length: int = 12) -> str:
    if not id_str:
        return "-"
    return id_str[:length]


def _fmt_dt(ts) -> str:
    if ts is None:
        return "-"
    if isinstance(ts, datetime):
        return ts.strftime("%Y-%m-%d %H:%M")
    return str(ts)


def _print_table(headers: list, rows: list) -> None:
    col_widths = [
        max(len(str(h)), max((len(str(row[i])) for row in rows), default=0))
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    click.echo(fmt.format(*headers))
    click.echo(sep)
    for row in rows:
        click.echo(fmt.format(*[str(v) for v in row]))


@click.group()
@click.version_option(version="0.1.0")
@click.option(
    "--project-dir",
    default=".",
    show_default=True,
    help="Project directory"
)
@click.pass_context
def cli(ctx, project_dir):
    """AgentTest - Record-replay regression testing for AI agents"""
    ctx.ensure_object(dict)
    ctx.obj["project_dir"] = project_dir


@cli.command(name="list")
@click.option(
    "--status",
    type=click.Choice(["in_progress", "completed", "failed"]),
    default=None,
    help="Filter by status"
)
@click.pass_context
def list_cmd(ctx, status):
    """List all recordings"""
    session = _make_session(ctx.obj["project_dir"])
    try:
        recordings = session.test_store.list_all_recordings(status=status)
        if not recordings:
            click.echo("No recordings found.")
            return
        rows = [
            (
                _fmt_id(r.recording_id),
                r.name,
                r.status.value,
                r.step_count,
                _fmt_dt(r.created_at)
            )
            for r in recordings
        ]
        _print_table(["ID", "Name", "Status", "Steps", "Created"], rows)
    finally:
        session.close()


@cli.command()
@click.argument("recording_id")
@click.pass_context
def show(ctx, recording_id):
    """Show details of a specific recording"""
    session = _make_session(ctx.obj["project_dir"])
    try:
        recording = session.get_recording(recording_id)
        if not recording:
            click.echo(f"Recording '{recording_id}' not found.", err=True)
            sys.exit(1)

        click.echo(f"Recording : {recording.recording_id}")
        click.echo(f"  Name    : {recording.name}")
        click.echo(f"  Status  : {recording.status.value}")
        click.echo(f"  Steps   : {recording.step_count}")
        click.echo(f"  Created : {_fmt_dt(recording.created_at)}")
        if recording.error:
            click.echo(f"  Error   : {recording.error}")

        details = session.get_recording_details(recording_id)
        if not details:
            click.echo("\nNo LLM call steps recorded.")
            return

        click.echo(f"\nLLM Call Steps ({len(details)}):")
        rows = [
            (
                d.step_index,
                d.model or "-",
                d.provider or "-",
                (d.fingerprint or "")[:8] or "-",
                d.duration_ms or 0
            )
            for d in details
        ]
        _print_table(["#", "Model", "Provider", "Fingerprint", "Duration ms"], rows)
    finally:
        session.close()


@cli.group()
def baseline():
    """Manage baselines"""
    pass


@baseline.command(name="list")
@click.pass_context
def baseline_list(ctx):
    """List all baselines"""
    session = _make_session(ctx.obj["project_dir"])
    try:
        tags = session.list_baselines()
        if not tags:
            click.echo("No baselines found.")
            return
        rows = [
            (
                t.tag_name[len("baseline/"):] if t.tag_name.startswith("baseline/") else t.tag_name,
                t.node_id,
                _fmt_id(t.metadata.get("recording_id")),
                _fmt_dt(t.created_at)
            )
            for t in tags
        ]
        _print_table(["Name", "Node ID", "Recording", "Created"], rows)
    finally:
        session.close()


@baseline.command(name="set")
@click.argument("name")
@click.argument("recording_id")
@click.pass_context
def baseline_set(ctx, name, recording_id):
    """Set a recording as a named baseline"""
    session = _make_session(ctx.obj["project_dir"])
    try:
        tag = session.set_baseline(name, recording_id)
        click.echo(f"Baseline '{name}' set → recording {_fmt_id(recording_id)}, node {tag.node_id}")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@baseline.command(name="delete")
@click.argument("name")
@click.pass_context
def baseline_delete(ctx, name):
    """Delete a named baseline"""
    session = _make_session(ctx.obj["project_dir"])
    try:
        deleted = session.delete_baseline(name)
        if deleted:
            click.echo(f"Baseline '{name}' deleted.")
        else:
            click.echo(f"Baseline '{name}' not found.")
    finally:
        session.close()


@cli.command()
@click.argument("comparison_id")
@click.pass_context
def diff(ctx, comparison_id):
    """Show details of a comparison"""
    session = _make_session(ctx.obj["project_dir"])
    try:
        result = session.get_comparison(comparison_id)
        if not result:
            click.echo(f"Comparison '{comparison_id}' not found.", err=True)
            sys.exit(1)

        if result.overall_pass:
            status_label = "PASS"
        elif result.has_regression and result.has_delta:
            status_label = "REGRESSION + DELTA"
        elif result.has_regression:
            status_label = "REGRESSION"
        else:
            status_label = "DELTA"
        click.echo(f"Comparison : {result.comparison_id}  [{status_label}]")
        click.echo(f"  Baseline : {_fmt_id(result.baseline_recording_id)}")
        click.echo(f"  Replay   : {_fmt_id(result.replay_recording_id)}")
        click.echo(f"  Matched  : {result.matched_steps} / {result.total_steps}")
        click.echo(f"  Diverged : {result.mismatched_steps}")
        click.echo(f"  Added    : {result.added_steps}")
        click.echo(f"  Removed  : {result.removed_steps}")
        click.echo(f"  Cascade  : {result.cascade_steps}")
        click.echo(f"  Regression: {result.regression_steps}")
        click.echo(f"  Delta     : {result.delta_steps}")

        if not result.step_comparisons:
            click.echo("\nNo step details available.")
            return

        click.echo(f"\nStep Details ({len(result.step_comparisons)}):")
        rows = [
            (
                sc.step_index,
                sc.status.value,
                sc.match_type.value if sc.match_type else "-",
                f"{sc.similarity_score:.2f}",
                (sc.diff_summary or "")[:50] or "-"
            )
            for sc in result.step_comparisons
        ]
        _print_table(["#", "Status", "Match Type", "Score", "Summary"], rows)
    finally:
        session.close()


@cli.command()
@click.option("--failed", is_flag=True, default=False, help="Show only failed comparisons")
@click.pass_context
def history(ctx, failed):
    """Show comparison history"""
    session = _make_session(ctx.obj["project_dir"])
    try:
        comparisons = session.test_store.list_all_comparisons(failed_only=failed)
        if not comparisons:
            click.echo("No comparisons found.")
            return
        rows = [
            (
                _fmt_id(c.comparison_id),
                _fmt_id(c.baseline_recording_id),
                _fmt_id(c.replay_recording_id),
                "PASS" if c.overall_pass else "FAIL",
                c.total_steps,
                c.matched_steps,
                _fmt_dt(c.created_at)
            )
            for c in comparisons
        ]
        _print_table(
            ["Comparison", "Baseline", "Replay", "Result", "Steps", "Matched", "Created"],
            rows
        )
    finally:
        session.close()


@cli.command()
@click.option("--mode", type=click.Choice(["locked", "selective", "full"]), default="locked", show_default=True)
@click.option("--tier", type=click.Choice(["always", "local", "ci-only", "all"]), default="all", show_default=True)
@click.argument("test_names", nargs=-1)
@click.pass_context
def replay(ctx, mode, tier, test_names):
    """Run agent tests in replay mode (default: locked)."""
    import subprocess
    args = ["python", "-m", "pytest", "--agenttest", f"--agenttest-mode={mode}", "-v"]
    if tier != "all":
        args += [f"--agenttest-tier={tier}"]
    if test_names:
        args += ["-k", " or ".join(test_names)]
    result = subprocess.run(args, cwd=ctx.obj["project_dir"])
    sys.exit(result.returncode)


@cli.command()
@click.argument("test_name", required=False)
@click.option("--baseline-name", default=None, help="Explicit baseline name to promote replay into")
@click.pass_context
def accept(ctx, test_name, baseline_name):
    """Promote the most recent replay recording to the new baseline."""
    session = _make_session(ctx.obj["project_dir"])
    try:
        recordings = session.test_store.list_all_recordings(status="completed")
        if test_name:
            recordings = [r for r in recordings if r.name == test_name or r.name.startswith(test_name)]
        if not recordings:
            click.echo("No completed recordings found to accept.", err=True)
            sys.exit(1)
        latest = recordings[0]
        resolved_baseline_name = baseline_name
        if not resolved_baseline_name:
            resolved_baseline_name = (latest.metadata or {}).get("baseline_name")
        if not resolved_baseline_name:
            resolved_baseline_name = latest.name
            for suffix in ("-replay", "_replay", "-test", "_test"):
                if resolved_baseline_name.endswith(suffix):
                    resolved_baseline_name = resolved_baseline_name[:-len(suffix)]
                    break
        if not resolved_baseline_name:
            click.echo("Unable to determine baseline name; pass --baseline-name.", err=True)
            sys.exit(1)
        tag = session.set_baseline(resolved_baseline_name, latest.recording_id)
        click.echo(f"Accepted recording '{latest.name}'")
        click.echo(f"  -> baseline '{resolved_baseline_name}' (node {tag.node_id})")
        click.echo(f"\nNext: git add .agentgit/ && git commit -m 'update baseline: {resolved_baseline_name}'")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command()
@click.pass_context
def status(ctx):
    """Show what changed vs baseline (no test run)."""
    session = _make_session(ctx.obj["project_dir"])
    try:
        baselines = session.list_baselines()
        if not baselines:
            click.echo("No baselines found. Run 'agenttest record --name=<test>' to create one.")
            return
        rows = []
        for tag in baselines:
            name = tag.tag_name.replace("baseline/", "")
            rec_id = tag.metadata.get("recording_id", "")
            latest = None
            if rec_id:
                latest = session.test_store.get_latest_comparison(
                    session.user_id, session.session_id, rec_id
                )
            if latest:
                if latest.overall_pass:
                    result_str = "PASS"
                elif latest.has_regression and latest.has_delta:
                    result_str = "REGRESSION + DELTA"
                elif latest.has_regression:
                    result_str = "REGRESSION"
                else:
                    result_str = "DELTA"
                detail = f"{latest.matched_steps}/{latest.total_steps} matched"
            else:
                result_str = "no comparisons"
                detail = "-"
            rows.append((name, _fmt_id(rec_id), result_str, detail))
        _print_table(["Baseline", "Recording", "Last Result", "Detail"], rows)
    finally:
        session.close()


@cli.command()
@click.option("--name", required=True, help="Test name to re-record")
@click.pass_context
def record(ctx, name):
    """Re-record a test from scratch (full mode, all live calls)."""
    import subprocess
    args = [
        "python",
        "-m",
        "pytest",
        "--agenttest",
        "--agenttest-record",
        "--agenttest-mode=full",
        "-v",
        "-k",
        name,
    ]
    result = subprocess.run(args, cwd=ctx.obj["project_dir"])
    sys.exit(result.returncode)


@cli.command(name="set-baseline")
@click.argument("name")
@click.argument("recording_id", required=False)
@click.pass_context
def set_baseline_cmd(ctx, name, recording_id):
    """Set a recording as a named baseline."""
    session = _make_session(ctx.obj["project_dir"])
    try:
        if not recording_id:
            recordings = session.test_store.list_all_recordings(status="completed")
            filtered = [r for r in recordings if r.name == name or r.name.startswith(name)]
            if not filtered:
                click.echo(f"No completed recording found for '{name}'", err=True)
                sys.exit(1)
            recording_id = filtered[0].recording_id
            click.echo(f"Using latest: {_fmt_id(recording_id)} ({filtered[0].name})")
        tag = session.set_baseline(name, recording_id)
        click.echo(f"Baseline '{name}' set -> recording {_fmt_id(recording_id)}, node {tag.node_id}")
        click.echo(f"\nNext: git add .agentgit/ && git commit -m 'add baseline: {name}'")
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        session.close()


@cli.command(name="pull-baseline")
@click.option("--from-run", required=True, help="GitHub Actions run ID")
@click.option("--artifact-name", default="agenttest-baseline", show_default=True)
@click.pass_context
def pull_baseline(ctx, from_run, artifact_name):
    """Pull baseline artifacts from a CI run (requires gh CLI)."""
    import subprocess
    import shutil
    from pathlib import Path as _Path
    project_dir = ctx.obj["project_dir"]
    ci_dir = _Path(project_dir) / ".agentgit-ci"
    agentgit_dir = _Path(project_dir) / ".agentgit"
    click.echo(f"Pulling baseline from CI run {from_run}...")
    result = subprocess.run(
        ["gh", "run", "download", from_run, "--name", artifact_name, "--dir", str(ci_dir)],
        cwd=project_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        click.echo(f"Failed: {result.stderr}", err=True)
        sys.exit(1)
    if not ci_dir.exists():
        click.echo("No artifacts found.", err=True)
        sys.exit(1)
    shutil.copytree(ci_dir, agentgit_dir, dirs_exist_ok=True)
    shutil.rmtree(ci_dir)
    click.echo("Baseline pulled -> .agentgit/")
    click.echo("\nNext: agenttest accept && git add .agentgit/ && git commit")


@cli.group()
def ci():
    """CI/CD integration commands"""
    pass


@ci.command(name="post-comment")
@click.option("--pr", default=None, help="PR number")
@click.pass_context
def post_comment(ctx, pr):
    """Post AgentTest results as a GitHub PR comment."""
    import os
    from agenttest.ci.reporter import format_pr_comment
    from agenttest.ci.github import post_pr_comment

    session = _make_session(ctx.obj["project_dir"])
    try:
        comparisons = session.test_store.list_all_comparisons()
        if not comparisons:
            click.echo("No comparisons to report.")
            return
        results = {}
        seen = set()
        for c in comparisons:
            if c.baseline_recording_id not in seen:
                baseline_rec = session.get_recording(c.baseline_recording_id)
                name = baseline_rec.name if baseline_rec else c.baseline_recording_id[:8]
                results[name] = c
                seen.add(c.baseline_recording_id)
        comment = format_pr_comment(results)
        click.echo(comment)
        pr_num = pr or os.environ.get("PR_NUMBER") or os.environ.get("GITHUB_PR_NUMBER")
        if pr_num:
            success = post_pr_comment(comment, pr_num)
            if success:
                click.echo("\nComment posted to PR.")
            else:
                click.echo("\nFailed to post comment.", err=True)
                sys.exit(1)
        else:
            click.echo("\n(No PR number — printed only)")
    finally:
        session.close()


if __name__ == "__main__":
    cli()
