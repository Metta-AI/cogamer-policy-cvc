"""Tests for the scenario harness.

Orchestration tests use a stubbed `run_episode_local`. A separate
scenario-marked smoke test (see `tests/scenarios/`) runs a real
mettagrid episode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from cvc_policy.scenarios import Scenario
from cvc_policy.scenarios.assertions import AssertResult
from cvc_policy.scenarios.harness import resolve_mission, run_scenario


class _FakeEpisodeResult:
    def __init__(self, steps: int = 3) -> None:
        self.steps = steps


def _stub_run_episode_local(**kwargs: Any) -> tuple[_FakeEpisodeResult, None]:
    # Write a minimal events.json so Run(run_dir) can load. Mettagrid's
    # real flow does this via CvCPolicy._on_episode_end; we short-circuit.
    init_kwargs = kwargs["policy_specs"][0].init_kwargs
    record_dir = Path(init_kwargs["record_dir"])
    (record_dir / "events.json").write_text(
        json.dumps(
            [
                {
                    "step": 0,
                    "agent": 0,
                    "stream": "py",
                    "type": "action",
                    "payload": {"role": "miner"},
                }
            ]
        )
    )
    return _FakeEpisodeResult(steps=3), None


def test_resolve_mission_machina_1() -> None:
    m = resolve_mission("machina_1", cogs=2)
    assert m.num_agents == 2


def test_resolve_mission_tutorial_variant() -> None:
    m = resolve_mission("tutorial.miner")
    assert "miner" in m._base_variants


def test_resolve_mission_unknown_raises() -> None:
    with pytest.raises(KeyError):
        resolve_mission("does_not_exist")


def test_run_scenario_writes_run_folder(tmp_path: Path) -> None:
    s = Scenario(
        name="my_test", tier=0, mission="machina_1", cogs=2, steps=3,
        assertions=[lambda run: AssertResult(name="dummy", passed=True)],
    )
    with patch(
        "cvc_policy.scenarios.harness.run_episode_local", side_effect=_stub_run_episode_local
    ):
        run = run_scenario(s, runs_root=tmp_path)
    assert run.run_dir.parent == tmp_path
    assert (run.run_dir / "events.json").exists()
    assert (run.run_dir / "result.json").exists()
    result = json.loads((run.run_dir / "result.json").read_text())
    assert result["scenario"] == "my_test"
    assert result["status"] == "passed"
    assert result["assertions"][0]["passed"] is True


def test_run_scenario_status_failed_when_assertion_fails(tmp_path: Path) -> None:
    s = Scenario(
        name="fail_test", tier=0, mission="machina_1", cogs=2, steps=3,
        assertions=[
            lambda run: AssertResult(name="x", passed=False, message="nope", failed_at_step=2)
        ],
    )
    with patch(
        "cvc_policy.scenarios.harness.run_episode_local", side_effect=_stub_run_episode_local
    ):
        run = run_scenario(s, runs_root=tmp_path)
    result = json.loads((run.run_dir / "result.json").read_text())
    assert result["status"] == "failed"


def test_run_scenario_applies_mission_overrides(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> tuple[_FakeEpisodeResult, None]:
        captured["env"] = kwargs["env"]
        return _stub_run_episode_local(**kwargs)

    s = Scenario(
        name="override_test", tier=0, mission="machina_1", cogs=2, steps=7,
        mission_overrides={"max_steps": 7},
    )
    with patch("cvc_policy.scenarios.harness.run_episode_local", side_effect=_capture):
        run_scenario(s, runs_root=tmp_path)
    assert captured["env"].game.max_steps == 7


def test_run_scenario_runs_setup_hook(tmp_path: Path) -> None:
    called = []

    def _setup(env_cfg: Any) -> None:
        called.append(env_cfg)
        env_cfg.game.agents[0].inventory.initial["miner"] = 1

    s = Scenario(
        name="setup_test", tier=0, mission="machina_1", cogs=1, steps=3,
        setup=_setup,
    )

    def _verify_inventory(**kwargs: Any) -> tuple[_FakeEpisodeResult, None]:
        assert kwargs["env"].game.agents[0].inventory.initial.get("miner") == 1
        return _stub_run_episode_local(**kwargs)

    with patch(
        "cvc_policy.scenarios.harness.run_episode_local", side_effect=_verify_inventory
    ):
        run_scenario(s, runs_root=tmp_path)
    assert len(called) == 1


def test_run_scenario_rejects_unknown_policy_kwargs(tmp_path: Path) -> None:
    s = Scenario(
        name="bad_kwargs", tier=0, mission="machina_1", cogs=1, steps=3,
        policy_kwargs={"not_a_real_kwarg": 42},
    )
    with pytest.raises(ValueError, match="unknown CvCPolicy kwarg"):
        run_scenario(s, runs_root=tmp_path)
