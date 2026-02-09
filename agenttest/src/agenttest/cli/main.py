"""
AgentTest CLI - Main entry point
"""
import click


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AgentTest - Record-replay regression testing for AI agents"""
    pass


@cli.command()
def list():
    """List recordings and baselines"""
    click.echo("agenttest list - Not yet implemented")


@cli.command()
@click.argument("recording_id")
def show(recording_id):
    """Show recording details"""
    click.echo(f"agenttest show {recording_id} - Not yet implemented")


@cli.group()
def baseline():
    """Manage baselines"""
    pass


@baseline.command(name="list")
def baseline_list():
    """List all baselines"""
    click.echo("agenttest baseline list - Not yet implemented")


@baseline.command(name="set")
@click.argument("name")
@click.argument("recording_id")
def baseline_set(name, recording_id):
    """Set a recording as baseline"""
    click.echo(f"agenttest baseline set {name} {recording_id} - Not yet implemented")


if __name__ == "__main__":
    cli()
