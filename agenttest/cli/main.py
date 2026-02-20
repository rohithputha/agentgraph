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

        status_label = "PASS" if result.overall_pass else "FAIL"
        click.echo(f"Comparison : {result.comparison_id}  [{status_label}]")
        click.echo(f"  Baseline : {_fmt_id(result.baseline_recording_id)}")
        click.echo(f"  Replay   : {_fmt_id(result.replay_recording_id)}")
        click.echo(f"  Matched  : {result.matched_steps} / {result.total_steps}")
        click.echo(f"  Diverged : {result.mismatched_steps}")
        click.echo(f"  Added    : {result.added_steps}")
        click.echo(f"  Removed  : {result.removed_steps}")
        click.echo(f"  Cascade  : {result.cascade_steps}")

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


if __name__ == "__main__":
    cli()
