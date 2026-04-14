"""Integration tests that drive CvCPolicyImpl.step_with_state with stubbed
programs and GameState to verify recorder emissions."""

from __future__ import annotations

from typing import Any

from cvc_policy.cogamer_policy import CvCAgentState, CvCPolicyImpl
from cvc_policy.proglet import Program
from cvc_policy.recorder import EventRecorder
from mettagrid.simulator import Action
from tests.conftest import _fake_policy_env_info


class _StubEngine:
    def __init__(self) -> None:
        self._llm_objective: str | None = None


class _StubGameState:
    def __init__(self) -> None:
        self.role: str = "miner"
        self.step_index: int = 0
        self.resource_bias: str | None = None
        self.engine = _StubEngine()
        self.finalized: list[str] = []

    def process_obs(self, obs: Any) -> None:
        self.step_index += 1

    def finalize_step(self, summary: str) -> None:
        self.finalized.append(summary)


def _make_impl(
    desired_role: str = "miner",
    summary: str = "mine_carbon",
    action_name: str = "noop",
) -> tuple[CvCPolicyImpl, CvCAgentState]:
    recorder = EventRecorder()
    programs = {
        "desired_role": Program(executor="code", fn=lambda gs: desired_role),
        "step": Program(
            executor="code", fn=lambda gs: (Action(action_name), summary)
        ),
        "summarize": Program(executor="code", fn=lambda gs: {"role": gs.role}),
    }
    impl = CvCPolicyImpl(
        _fake_policy_env_info(),
        agent_id=0,
        programs=programs,
        llm_client=None,
        recorder=recorder,
    )
    state = CvCAgentState(game_state=_StubGameState())  # type: ignore[arg-type]
    return impl, state


def test_step_emits_action_event_per_tick():
    impl, state = _make_impl()
    for _ in range(3):
        impl.step_with_state(object(), state)
    action_events = [e for e in impl._recorder.events if e["type"] == "action"]
    assert len(action_events) == 3
    assert all(e["payload"].get("role") == "miner" for e in action_events)
    assert all(e["agent"] == 0 for e in action_events)
    assert all(e["stream"] == "py" for e in action_events)


def test_role_change_event_fires_on_transition():
    impl, state = _make_impl(desired_role="aligner")
    # initial role on stub gs is "miner"
    impl.step_with_state(object(), state)
    events = impl._recorder.events
    changes = [e for e in events if e["type"] == "role_change"]
    assert len(changes) == 1
    assert changes[0]["payload"] == {"from": "miner", "to": "aligner"}
    assert changes[0]["agent"] == 0


def test_no_role_change_event_when_role_stable():
    impl, state = _make_impl(desired_role="miner")
    impl.step_with_state(object(), state)
    impl.step_with_state(object(), state)
    assert [e for e in impl._recorder.events if e["type"] == "role_change"] == []


def test_recorder_step_is_set_each_tick():
    impl, state = _make_impl()
    impl.step_with_state(object(), state)
    impl.step_with_state(object(), state)
    steps = sorted({e["step"] for e in impl._recorder.events})
    assert steps == [1, 2]
