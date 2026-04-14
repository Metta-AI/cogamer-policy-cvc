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


def test_view_stub_exits_zero() -> None:
    result = CliRunner().invoke(app, ["view", "any-run-id"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output.lower()


def test_test_cov_stub_exits_zero() -> None:
    result = CliRunner().invoke(app, ["test-cov"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output.lower()
