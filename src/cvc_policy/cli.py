"""`cgp` — CvC policy diagnostic CLI.

Top-level typer app with subcommand groups. Most commands are stubbed
and filled in across Batch 2 (scenarios, play) and later batches
(view, test-cov).
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="cgp",
    help="CvC policy diagnostic CLI: scenarios, runs, play, and reports.",
    no_args_is_help=True,
    add_completion=False,
)

scenario_app = typer.Typer(help="Scenario registry + runner.", no_args_is_help=True)
app.add_typer(scenario_app, name="scenario")


@scenario_app.command("list")
def scenario_list() -> None:
    """List registered scenarios. (Registry is empty until Task 2.2.)"""
    typer.echo("(no scenarios registered)")


@app.command("view")
def view(run_id: str) -> None:
    """Render an HTML report for a run. Stub until Batch 3."""
    typer.echo(f"view {run_id}: not yet implemented (Batch 3)")


@app.command("runs")
def runs() -> None:
    """List past runs, most recent first. Stub until Batch 3."""
    typer.echo("runs: not yet implemented (Batch 3)")


@app.command("play")
def play() -> None:
    """Play a mission ad-hoc with overrides. Stub until Task 2.12."""
    typer.echo("play: not yet implemented (Task 2.12)")


@app.command("test-cov")
def test_cov() -> None:
    """Run pytest with coverage. Stub until Batch 4."""
    typer.echo("test-cov: not yet implemented (Batch 4)")


if __name__ == "__main__":
    app()
