"""`cgp` — CvC policy diagnostic CLI.

Top-level typer app with subcommand groups. Most commands are
implemented across Batch 2; view + runs + test-cov are stubs until
later batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from cvc_policy.scenarios import registry
from cvc_policy.scenarios.harness import run_scenario

app = typer.Typer(
    name="cgp",
    help="CvC policy diagnostic CLI: scenarios, runs, play, and reports.",
    no_args_is_help=True,
    add_completion=False,
)

scenario_app = typer.Typer(help="Scenario registry + runner.", no_args_is_help=True)
app.add_typer(scenario_app, name="scenario")


def _load_all_scenarios() -> None:
    """Import every scenario case module so the registry is populated.

    Called lazily by CLI commands that need the full registry. Each
    import has a side effect of registering the scenario via the
    `@scenario` decorator.
    """
    # Imports are local so scenario modules aren't loaded just to
    # print `--help` on an unrelated subcommand.
    import cvc_policy.scenarios.cases.empty_extractor_skipped  # noqa: F401
    import cvc_policy.scenarios.cases.exploration_small  # noqa: F401
    import cvc_policy.scenarios.cases.mining_discovers_cap  # noqa: F401
    import cvc_policy.scenarios.cases.mining_trip_efficiency  # noqa: F401
    import cvc_policy.scenarios.cases.smoke  # noqa: F401


@scenario_app.command("list")
def scenario_list() -> None:
    """List registered scenarios grouped by tier."""
    _load_all_scenarios()
    reg = registry()
    if not reg:
        typer.echo("(no scenarios registered)")
        return
    current_tier: int | None = None
    for name, s in reg.items():
        if s.tier != current_tier:
            typer.echo(f"tier {s.tier}:")
            current_tier = s.tier
        typer.echo(f"  {name}  ({s.mission}, cogs={s.cogs}, steps={s.steps})")


@scenario_app.command("run")
def scenario_run(
    name: str,
    steps: Optional[int] = typer.Option(None, "--steps", help="Override scenario steps."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override scenario seed."),
    no_assert: bool = typer.Option(False, "--no-assert", help="Skip assertions."),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root", help="Run folder root."),
) -> None:
    """Run a single scenario by name."""
    _load_all_scenarios()
    reg = registry()
    if name not in reg:
        typer.echo(f"unknown scenario: {name}")
        typer.echo(f"available: {', '.join(reg.keys())}")
        raise typer.Exit(code=2)
    s = reg[name]
    if seed is not None:
        s = _replace_seed(s, seed)
    run = run_scenario(
        s,
        steps_override=steps,
        runs_root=runs_root,
        skip_assertions=no_assert,
    )
    status = run.result.get("status", "unknown")
    typer.echo(f"{name}: {status} ({run.run_dir})")
    for a in run.result.get("assertions", []):
        mark = "PASS" if a["passed"] else "FAIL"
        typer.echo(f"  [{mark}] {a['name']}: {a['message']}")
    if status != "passed":
        raise typer.Exit(code=1)


@scenario_app.command("run-all")
def scenario_run_all(
    tier: Optional[int] = typer.Option(None, "--tier", help="Filter by tier."),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root", help="Run folder root."),
) -> None:
    """Run every registered scenario (optionally filtered by tier)."""
    _load_all_scenarios()
    reg = registry()
    selected = [s for s in reg.values() if tier is None or s.tier == tier]
    if not selected:
        typer.echo("(no scenarios matched)")
        raise typer.Exit(code=0)
    failures: list[str] = []
    for s in selected:
        run = run_scenario(s, runs_root=runs_root)
        status = run.result.get("status", "unknown")
        typer.echo(f"{s.name}: {status}")
        if status != "passed":
            failures.append(s.name)
    typer.echo(
        f"\n{len(selected) - len(failures)}/{len(selected)} passed"
        + (f" — failed: {', '.join(failures)}" if failures else "")
    )
    if failures:
        raise typer.Exit(code=1)


def _replace_seed(scenario_obj, seed: int):
    import dataclasses

    return dataclasses.replace(scenario_obj, seed=seed)


@app.command("view")
def view(
    run: str = typer.Argument(..., help="Run id (under --runs-root) or path."),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
    no_open: bool = typer.Option(False, "--no-open", help="Do not launch browser."),
) -> None:
    """Render and open the HTML report for a run."""
    import webbrowser

    from cvc_policy.viewer import render

    candidate = Path(run)
    if candidate.is_dir():
        run_dir = candidate
    else:
        run_dir = runs_root / run
        # Reject path traversal: the resolved run_dir must sit under
        # the resolved runs_root.
        resolved = run_dir.resolve()
        root_resolved = runs_root.resolve()
        try:
            resolved.relative_to(root_resolved)
        except ValueError:
            raise typer.BadParameter(
                f"invalid run id (path traversal outside {runs_root}): {run}"
            )
    if not run_dir.is_dir():
        typer.echo(f"no such run: {run_dir}")
        raise typer.Exit(code=2)
    out = render(run_dir)
    typer.echo(f"wrote {out}")
    if not no_open:
        webbrowser.open(str(out.resolve()))


@app.command("runs")
def runs(
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
) -> None:
    """List past runs, most recent first."""
    import json as _json

    if not runs_root.is_dir():
        typer.echo("(no runs)")
        return
    entries = [p for p in runs_root.iterdir() if p.is_dir()]
    entries.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not entries:
        typer.echo("(no runs)")
        return
    typer.echo(f"{'run_id':<40} {'scenario':<30} {'status':<10} {'duration':>8}")
    for p in entries:
        result_path = p / "result.json"
        if result_path.exists():
            r = _json.loads(result_path.read_text())
            scen = str(r.get("scenario") or "")
            status = str(r.get("status") or "")
            dur = float(r.get("duration_s") or 0.0)
        else:
            scen, status, dur = "", "(missing)", 0.0
        typer.echo(f"{p.name:<40} {scen:<30} {status:<10} {dur:>7.2f}s")


@app.command("play")
def play(
    mission: str = typer.Option(..., "-m", "--mission", help="Mission name (e.g. machina_1)."),
    variant: list[str] = typer.Option(
        [], "-v", "--variant", help="Variant name; repeat flag to add multiple."
    ),
    cogs: int = typer.Option(1, "-c", "--cogs", help="Number of cogs."),
    steps: int = typer.Option(500, "-s", "--steps", help="Max episode steps."),
    seed: int = typer.Option(42, "--seed"),
    override: list[str] = typer.Option(
        [], "--override", help="Mission field override KEY=VALUE (repeatable)."
    ),
    variant_override: list[str] = typer.Option(
        [], "--variant-override", help="Variant field override VARIANT.KEY=VALUE (repeatable)."
    ),
    policy_args: list[str] = typer.Option(
        [], "--policy-args", help="Policy kwarg KEY=VALUE (repeatable)."
    ),
    record: bool = typer.Option(
        True, "--record/--no-record", help="Write a run folder under --runs-root."
    ),
    runs_root: Path = typer.Option(Path("runs"), "--runs-root"),
) -> None:
    """Play a mission ad-hoc with optional mission/variant/policy overrides."""
    from cvc_policy.overrides import parse_override, parse_variant_override
    from cvc_policy.scenarios import Scenario

    mission_overrides: dict[str, object] = {}
    for spec in override:
        k, v = parse_override(spec)
        mission_overrides[k] = v

    v_overrides: dict[str, dict[str, object]] = {}
    for spec in variant_override:
        vname, k, val = parse_variant_override(spec)
        v_overrides.setdefault(vname, {})[k] = val

    policy_kwargs: dict[str, object] = {}
    for spec in policy_args:
        k, val = parse_override(spec)
        policy_kwargs[k] = val

    synthetic = Scenario(
        name="manual",
        tier=-1,
        mission=mission,
        variants=tuple(variant),
        cogs=cogs,
        steps=steps,
        seed=seed,
        policy_kwargs=policy_kwargs,
        mission_overrides=mission_overrides,
        variant_overrides=v_overrides,
        assertions=[],
    )
    if not record:
        import tempfile

        runs_root = Path(tempfile.mkdtemp(prefix="cgp-play-"))
    run = run_scenario(synthetic, runs_root=runs_root)
    typer.echo(f"manual run: {run.run_dir}")
    typer.echo(
        f"steps: {run.result.get('steps')}  "
        f"duration: {run.result.get('duration_s') or 0.0:.2f}s"
    )


@app.command("test-cov")
def test_cov() -> None:
    """Run pytest with coverage. Stub until Batch 4."""
    typer.echo("test-cov: not yet implemented (Batch 4)")


if __name__ == "__main__":
    app()
