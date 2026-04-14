"""Tests for the `cgp` CLI skeleton."""

from __future__ import annotations

from typer.testing import CliRunner

from cvc_policy.cli import app


def test_cli_help_lists_subcommands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    # Top-level should mention each subcommand group.
    for cmd in ("scenario", "view", "play", "runs", "test-cov"):
        assert cmd in result.output


def test_scenario_list_runs_without_error() -> None:
    result = CliRunner().invoke(app, ["scenario", "list"])
    assert result.exit_code == 0


def test_view_errors_on_missing_run(tmp_path) -> None:
    result = CliRunner().invoke(
        app, ["view", "does-not-exist", "--runs-root", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "no such run" in result.output.lower()


def test_test_cov_invokes_pytest_with_coverage(monkeypatch) -> None:
    import subprocess

    calls: list[list[str]] = []

    class _FakeCompleted:
        returncode = 0

    def fake_run(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return _FakeCompleted()

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CliRunner().invoke(app, ["test-cov"])
    assert result.exit_code == 0
    assert len(calls) == 1
    cmd = calls[0]
    assert "pytest" in cmd
    assert "--cov=cvc_policy" in cmd
    assert "--cov-report=term-missing" in cmd
    assert "--cov-report=xml" in cmd


def test_test_cov_propagates_nonzero_exit(monkeypatch) -> None:
    import subprocess

    class _FakeCompleted:
        returncode = 3

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _FakeCompleted())
    result = CliRunner().invoke(app, ["test-cov"])
    assert result.exit_code == 3


def test_scenario_list_shows_registered_scenarios() -> None:
    # Load all 5 Batch 2 scenarios.
    import cvc_policy.scenarios.cases.empty_extractor_skipped  # noqa: F401
    import cvc_policy.scenarios.cases.exploration_small  # noqa: F401
    import cvc_policy.scenarios.cases.mining_discovers_cap  # noqa: F401
    import cvc_policy.scenarios.cases.mining_trip_efficiency  # noqa: F401
    import cvc_policy.scenarios.cases.smoke  # noqa: F401

    result = CliRunner().invoke(app, ["scenario", "list"])
    assert result.exit_code == 0
    for name in (
        "smoke_machina1_runs",
        "exploration_small",
        "mining_discovers_cap",
        "mining_trip_efficiency",
        "empty_extractor_skipped",
    ):
        assert name in result.output


def test_scenario_run_invokes_harness(tmp_path, monkeypatch) -> None:
    """scenario run <name> should call run_scenario with that name."""
    import cvc_policy.scenarios.cases.smoke  # noqa: F401
    from cvc_policy.scenarios._run import Run
    import cvc_policy.cli as cli_mod

    called = {}

    def fake_run_scenario(scenario, **kwargs):
        called["scenario_name"] = scenario.name
        run_dir = kwargs["runs_root"] / "fake-run"
        run_dir.mkdir(parents=True)
        (run_dir / "events.json").write_text("[]")
        (run_dir / "result.json").write_text(
            '{"run_id": "fake-run", "status": "passed", "assertions": []}'
        )
        return Run(run_dir)

    monkeypatch.setattr(cli_mod, "run_scenario", fake_run_scenario)
    result = CliRunner().invoke(
        app, ["scenario", "run", "smoke_machina1_runs", "--runs-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert called["scenario_name"] == "smoke_machina1_runs"
    assert "passed" in result.output.lower()


def test_scenario_run_unknown_name_exits_nonzero(tmp_path) -> None:
    result = CliRunner().invoke(
        app, ["scenario", "run", "no_such_scenario", "--runs-root", str(tmp_path)]
    )
    assert result.exit_code != 0


def test_play_builds_scenario_from_cli_args(tmp_path, monkeypatch) -> None:
    from cvc_policy.scenarios._run import Run
    import cvc_policy.cli as cli_mod

    captured = {}

    def fake_run_scenario(scenario, **kwargs):
        captured["scenario"] = scenario
        captured["kwargs"] = kwargs
        run_dir = kwargs["runs_root"] / "manual-run"
        run_dir.mkdir(parents=True)
        (run_dir / "events.json").write_text("[]")
        (run_dir / "result.json").write_text(
            '{"run_id": "manual-run", "status": "passed", "assertions": [], '
            '"steps": 10, "duration_s": 0.5}'
        )
        return Run(run_dir)

    monkeypatch.setattr(cli_mod, "run_scenario", fake_run_scenario)
    result = CliRunner().invoke(
        app,
        [
            "play",
            "-m", "machina_1",
            "-c", "2",
            "-s", "30",
            "--seed", "7",
            "--override", "max_steps=30",
            "--policy-args", "log_py=true",
            "--runs-root", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    s = captured["scenario"]
    assert s.mission == "machina_1"
    assert s.cogs == 2
    assert s.steps == 30
    assert s.seed == 7
    assert s.mission_overrides == {"max_steps": 30}
    assert s.policy_kwargs == {"log_py": True}


def test_play_tolerates_missing_duration_s(tmp_path, monkeypatch) -> None:
    """Old result.json files (or harness bugs) may lack duration_s;
    cgp play must not raise TypeError formatting a None."""
    from cvc_policy.scenarios._run import Run
    import cvc_policy.cli as cli_mod

    def fake_run_scenario(scenario, **kwargs):
        run_dir = kwargs["runs_root"] / "nodur-run"
        run_dir.mkdir(parents=True)
        (run_dir / "events.json").write_text("[]")
        # result.json is missing duration_s.
        (run_dir / "result.json").write_text(
            '{"run_id": "nodur-run", "status": "passed", "assertions": [], "steps": 10}'
        )
        return Run(run_dir)

    monkeypatch.setattr(cli_mod, "run_scenario", fake_run_scenario)
    result = CliRunner().invoke(
        app,
        ["play", "-m", "machina_1", "-c", "1", "-s", "10", "--runs-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "duration: 0.00s" in result.output


def test_scenario_run_all_tier_filter(tmp_path, monkeypatch) -> None:
    import cvc_policy.scenarios.cases.smoke  # noqa: F401
    import cvc_policy.scenarios.cases.exploration_small  # noqa: F401
    from cvc_policy.scenarios._run import Run
    import cvc_policy.cli as cli_mod

    ran: list[str] = []

    def fake_run_scenario(scenario, **kwargs):
        ran.append(scenario.name)
        run_dir = kwargs["runs_root"] / f"fake-{scenario.name}"
        run_dir.mkdir(parents=True)
        (run_dir / "events.json").write_text("[]")
        (run_dir / "result.json").write_text(
            '{"run_id": "x", "status": "passed", "assertions": []}'
        )
        return Run(run_dir)

    monkeypatch.setattr(cli_mod, "run_scenario", fake_run_scenario)
    # tier 0 only: should run just smoke_machina1_runs
    result = CliRunner().invoke(
        app, ["scenario", "run-all", "--tier", "0", "--runs-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "smoke_machina1_runs" in ran
    assert "exploration_small" not in ran
