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
from cvc_policy.logcfg import LogConfig
from cvc_policy.programs import all_programs
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
        log: LogConfig | None = None,
    ) -> None:
        self._policy_env_info = policy_env_info
        self._agent_id = agent_id
        self._programs = programs
        self._llm_client = llm_client
        self._game_id = game_id
        self._log = log if log is not None else LogConfig("")

    def initial_agent_state(self) -> CvCAgentState:
        gs = GameState(
            self._policy_env_info,
            agent_id=self._agent_id,
        )
        state = CvCAgentState(game_state=gs)
        # Wire log_to_llm onto GameState so code programs can push events.
        gs.log_to_llm = lambda event: _log(state.log_queue, event)  # type: ignore[attr-defined]
        if self._llm_client is not None:
            state.worker = LLMWorker(self._llm_client, self._agent_id, state, log=self._log)
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
        prev_role = gs.role
        gs.role = self._invoke_sync("desired_role", gs)
        # LLM role override wins over the heuristic role choice (soft hint).
        if state.llm_role_override is not None:
            gs.role = state.llm_role_override
        if self._log.enabled("py") and gs.role != prev_role:
            self._log.log(
                "py",
                f"a{self._agent_id} step={gs.step_index} role {prev_role}->{gs.role}",
            )

        action, summary = self._invoke_sync("step", gs)
        gs.finalize_step(summary)
        if self._log.enabled("py"):
            self._log.log(
                "py",
                f"a{self._agent_id} step={gs.step_index} role={gs.role} {summary}",
            )

        # Heartbeat: feed the LLM a periodic snapshot.
        if state.worker is not None and gs.step_index % _HEARTBEAT_EVERY == 0:
            snapshot = self._invoke_sync("summarize", gs)
            _log(state.log_queue, {"kind": "heartbeat", **snapshot})

        return action, state


def _log(q: queue.Queue, event: dict) -> None:
    """Best-effort enqueue. Drops the event if the queue is full."""
    try:
        q.put_nowait(event)
    except queue.Full:
        pass


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
        **kwargs: Any,
    ):
        super().__init__(policy_env_info, device=device, **kwargs)
        self._programs = programs or all_programs()
        self._agent_policies: dict[int, StatefulAgentPolicy[CvCAgentState]] = {}
        self._llm_client: Any | None = None
        self._episode_start = time.time()
        self._game_id = kwargs.get("game_id", f"game_{int(time.time())}")
        self._log = LogConfig(log)
        if self._log.enabled("py") or self._log.enabled("llm"):
            self._log.log("py", f"policy init streams={self._log!r}")
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
                log=self._log,
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
