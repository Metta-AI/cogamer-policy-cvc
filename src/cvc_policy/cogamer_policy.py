"""CogamerPolicy: program-table-driven CvC policy.

Dispatches through a flat program table operating on GameState.
Each agent is fully independent — no shared state between agents.

Architecture:
  CvCPolicy (MultiAgentPolicy)
    └─ StatefulAgentPolicy[CvCAgentState]  (one per agent)
         └─ CvCPolicyImpl (StatefulPolicyImpl)
              └─ GameState (observation processing + mutable state)
              └─ Program table (step/heal/retreat/mine/align/scramble/explore)
              └─ LLMWorker thread (per-agent, episode-long Anthropic session,
                 reads logs from the agent's queue and patches strategic knobs)
"""

from __future__ import annotations

import json
import os
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cvc_policy.game_state import GameState
from cvc_policy.llm_worker import LLMWorker
from cvc_policy.programs import all_programs
from cvc_policy.recorder import EventRecorder, fmt
from mettagrid.policy.policy import MultiAgentPolicy, StatefulAgentPolicy, StatefulPolicyImpl
from mettagrid.policy.policy_env_interface import PolicyEnvInterface
from mettagrid.simulator import Action
from mettagrid.simulator.interface import AgentObservation

try:
    from cvc_policy.proglet import Program
except ImportError:
    Program = None  # type: ignore[assignment,misc]

_HEARTBEAT_EVERY = 200
_QUEUE_MAX = 1000
_TRACE_DIR = os.environ.get("CVC_TRACE_DIR", "/tmp/cvc-trace")


@dataclass
class CvCAgentState:
    """All mutable state for one agent."""

    game_state: GameState | None = None
    llm_latencies: list[float] = field(default_factory=list)
    resource_bias_from_llm: str | None = None
    llm_role_override: str | None = None
    llm_objective: str | None = None
    llm_log: list[dict[str, Any]] = field(default_factory=list)
    snapshot_log: list[dict[str, Any]] = field(default_factory=list)
    experience: list[dict[str, Any]] = field(default_factory=list)
    log_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=_QUEUE_MAX))
    worker: LLMWorker | None = None


class CvCPolicyImpl(StatefulPolicyImpl[CvCAgentState]):
    """Per-agent decision logic using the program table."""

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        agent_id: int,
        programs: dict[str, Program],
        llm_client: Any | None = None,
        game_id: str = "",
        recorder: EventRecorder | None = None,
    ) -> None:
        self._policy_env_info = policy_env_info
        self._agent_id = agent_id
        self._programs = programs
        self._llm_client = llm_client
        self._game_id = game_id
        self._recorder = recorder if recorder is not None else EventRecorder()
        self._infos: dict[str, Any] = {}

    def initial_agent_state(self) -> CvCAgentState:
        gs = GameState(
            self._policy_env_info,
            agent_id=self._agent_id,
        )
        # Wire cap-discovery events into the recorder.
        agent_id = self._agent_id
        recorder = self._recorder

        def _on_cap_discovery(sig: tuple[str, ...], cap: int) -> None:
            recorder.emit(
                type="cap_discovered",
                agent=agent_id,
                stream="py",
                payload={"gear_sig": list(sig), "cap": cap},
            )

        gs.engine._cargo_cap._on_discovery = _on_cap_discovery
        state = CvCAgentState(game_state=gs)
        # Wire log_to_llm onto GameState so code programs can push events.
        gs.log_to_llm = lambda event: _log(state.log_queue, event)  # type: ignore[attr-defined]
        if self._llm_client is not None:
            state.worker = LLMWorker(
                self._llm_client, self._agent_id, state, recorder=self._recorder
            )
            state.worker.start()
        return state

    def _invoke_sync(self, name: str, *args: Any) -> Any:
        prog = self._programs[name]
        if prog.executor == "code" and prog.fn is not None:
            return prog.fn(*args)
        raise ValueError(f"Cannot sync-invoke {name} (executor={prog.executor})")

    def step_with_state(self, obs: AgentObservation, state: CvCAgentState) -> tuple[Action, CvCAgentState]:
        gs = state.game_state
        assert gs is not None

        # Apply any LLM-set knobs before action selection.
        if state.resource_bias_from_llm is not None:
            gs.resource_bias = state.resource_bias_from_llm
        if state.llm_objective is not None and hasattr(gs.engine, "_llm_objective"):
            gs.engine._llm_objective = state.llm_objective

        gs.process_obs(obs)
        self._recorder.set_step(gs.step_index)
        prev_role = gs.role
        gs.role = self._invoke_sync("desired_role", gs)
        # LLM role override wins over the heuristic role choice (soft hint).
        if state.llm_role_override is not None:
            gs.role = state.llm_role_override
        if gs.role != prev_role:
            self._recorder.emit(
                type="role_change",
                agent=self._agent_id,
                stream="py",
                payload={"from": prev_role, "to": gs.role},
            )

        action, summary = self._invoke_sync("step", gs)
        gs.finalize_step(summary)
        self._recorder.emit(
            type="action",
            agent=self._agent_id,
            stream="py",
            payload={"role": gs.role, "summary": summary},
        )
        target_kind = getattr(gs.engine, "_current_target_kind", None)
        target_pos = getattr(gs.engine, "_current_target_position", None)
        if target_kind and target_pos is not None:
            self._recorder.emit(
                type="target",
                agent=self._agent_id,
                stream="py",
                payload={"kind": target_kind, "pos": list(target_pos)},
            )

        # Heartbeat: feed the LLM a periodic snapshot and record it.
        if gs.step_index > 0 and gs.step_index % _HEARTBEAT_EVERY == 0:
            snapshot = self._invoke_sync("summarize", gs)
            self._recorder.emit(
                type="heartbeat",
                agent=self._agent_id,
                stream="py",
                payload=dict(snapshot),
            )
            if state.worker is not None:
                _log(state.log_queue, {"kind": "heartbeat", **snapshot})

        # Surface this tick's events (for this agent) via policyInfos so
        # mettagrid persists them in the replay. Also include team/global
        # events (agent=None) so they appear on every agent's stream.
        tick_events = [
            e
            for e in self._recorder.events_for_step(gs.step_index)
            if e["agent"] == self._agent_id or e["agent"] is None
        ]
        summary_lines = [fmt(e) for e in tick_events]
        self._infos = {
            "events": tick_events,
            "summary": "\n".join(summary_lines),
        }

        return action, state


def _log(q: queue.Queue, event: dict) -> None:
    """Best-effort enqueue. Drops the event if the queue is full."""
    try:
        q.put_nowait(event)
    except queue.Full:
        pass


def _truthy(value: Any) -> bool:
    """Handles both real bools and CLI-style string values ('1','true','yes')."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y", "t"}


class CvCPolicy(MultiAgentPolicy):
    """Top-level CvC policy. Spawns one LLMWorker thread per agent."""

    short_names = ["cvc", "cvc-policy"]
    minimum_action_timeout_ms = 30_000

    def __init__(
        self,
        policy_env_info: PolicyEnvInterface,
        device: str = "cpu",
        programs: dict[str, Program] | None = None,
        log: str | None = None,
        log_py: Any = None,
        log_llm: Any = None,
        game_id: str | None = None,
        record_dir: str | None = None,
        **kwargs: Any,
    ):
        # Fail loudly on unknown kwargs — silent swallowing (mettagrid's
        # MultiAgentPolicy takes **kwargs) masked the log-py=1 bug.
        if kwargs:
            raise TypeError(
                f"CvCPolicy got unknown kwarg(s): {sorted(kwargs)}. "
                "Known kwargs: device, programs, log, log_py, log_llm, "
                "game_id, record_dir."
            )
        super().__init__(policy_env_info, device=device)
        self._programs = programs or all_programs()
        self._agent_policies: dict[int, StatefulAgentPolicy[CvCAgentState]] = {}
        self._llm_client: Any | None = None
        self._episode_start = time.time()
        self._game_id = game_id if game_id is not None else f"game_{int(time.time())}"
        self._record_dir = record_dir
        streams: set[str] = set()
        if log:
            for part in str(log).split("+"):
                part = part.strip().lower()
                if part == "all":
                    streams.update({"py", "llm"})
                elif part in {"py", "llm"}:
                    streams.add(part)
        if _truthy(log_py):
            streams.add("py")
        if _truthy(log_llm):
            streams.add("llm")
        self._recorder = EventRecorder(
            stderr_streams=streams, record_dir=record_dir
        )
        self._init_llm()

        import atexit

        atexit.register(self._on_episode_end)

    def _init_llm(self) -> None:
        api_key = os.environ.get("COGORA_ANTHROPIC_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return
        try:
            import anthropic

            self._llm_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            pass

    @property
    def programs(self) -> dict[str, Program]:
        return self._programs

    def agent_policy(self, agent_id: int) -> StatefulAgentPolicy[CvCAgentState]:
        if agent_id not in self._agent_policies:
            impl = CvCPolicyImpl(
                self._policy_env_info,
                agent_id,
                programs=self._programs,
                llm_client=self._llm_client,
                game_id=self._game_id,
                recorder=self._recorder,
            )
            self._agent_policies[agent_id] = StatefulAgentPolicy(
                impl,
                self._policy_env_info,
                agent_id=agent_id,
            )
        return self._agent_policies[agent_id]

    def reset(self) -> None:
        if self._agent_policies:
            self._on_episode_end()
        self._episode_start = time.time()
        for p in self._agent_policies.values():
            p.reset()

    def _stop_workers(self) -> None:
        for wrapper in self._agent_policies.values():
            st: CvCAgentState | None = getattr(wrapper, "_state", None)
            if st is not None and st.worker is not None:
                st.worker.stop(timeout=2.0)
                st.worker = None

    def _on_episode_end(self) -> None:
        self._stop_workers()
        self._write_trace()
        if self._record_dir:
            self._recorder.flush_json(Path(self._record_dir) / "events.json")

    def _write_trace(self) -> None:
        """Write LLM↔Python communication trace to disk for analysis."""
        trace_dir = Path(_TRACE_DIR)
        trace_dir.mkdir(parents=True, exist_ok=True)

        all_llm: list[dict] = []
        agents_data: dict[str, Any] = {}
        for aid, wrapper in self._agent_policies.items():
            st: CvCAgentState | None = getattr(wrapper, "_state", None)
            if st is None:
                continue
            gs = st.game_state
            agents_data[str(aid)] = {
                "steps": gs.step_index if gs else 0,
                "llm_calls": len(st.llm_log),
                "final_resource_bias": st.resource_bias_from_llm,
                "final_role_override": st.llm_role_override,
                "final_objective": st.llm_objective,
            }
            for entry in st.llm_log:
                all_llm.append({"agent": aid, **entry})

        trace = {
            "game_id": self._game_id,
            "duration_s": round(time.time() - self._episode_start, 1),
            "agents": agents_data,
            "llm_trace": all_llm,
        }

        path = trace_dir / f"{self._game_id}.json"
        path.write_text(json.dumps(trace, indent=2, default=str))
