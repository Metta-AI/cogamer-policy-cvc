"""Per-agent LLM coach worker.

Each agent owns one LLMWorker running in a dedicated thread. The worker holds a
single episode-long Anthropic conversation. It loops forever (no sleep) calling
`read_recent_logs` (blocks up to 1s on an empty queue) to pull events from the
Python tick loop, and `patch` to write strategic knobs back onto the agent's
state. The Python tick loop only reads those knobs — it never waits on the LLM.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import TYPE_CHECKING, Any

from cvc_policy.recorder import EventRecorder

if TYPE_CHECKING:
    from cvc_policy.cogamer_policy import CvCAgentState


_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 400
_READ_TIMEOUT_S = 1.0
_HISTORY_TRIM_AT = 120
_HISTORY_KEEP_TAIL = 40

_SYSTEM = (
    "You are the strategic coach for a single agent in the CvC game. "
    "The agent picks its own low-level actions. You only steer three knobs "
    "by calling `patch`: resource_bias (which element to prioritize mining), "
    "role (miner/aligner/scrambler — null to leave unchanged), "
    "and objective (expand/defend/economy_bootstrap — null to leave unchanged).\n"
    "\n"
    "Work in a loop: call `read_recent_logs` to pull recent game events, "
    "reason briefly, then call `patch` when the state should change. Repeat "
    "continuously for the whole episode. `read_recent_logs` blocks up to 1s "
    "when the queue is empty — just call it again if it returns no events."
)

_TOOLS = [
    {
        "name": "read_recent_logs",
        "description": (
            "Drain recent log events from the agent's event queue. Blocks up to "
            "1 second waiting for the first event; returns immediately once any "
            "events are available, up to max_events total."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_events": {
                    "type": "integer",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 100,
                },
            },
        },
    },
    {
        "name": "patch",
        "description": (
            "Patch the agent's strategic knobs. All fields are optional; only "
            "provided fields are updated. Call whenever you have a new strategic "
            "decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "resource_bias": {
                    "type": "string",
                    "enum": ["carbon", "oxygen", "germanium", "silicon"],
                },
                "role": {
                    "type": "string",
                    "enum": ["miner", "aligner", "scrambler"],
                },
                "objective": {
                    "type": "string",
                    "enum": ["expand", "defend", "economy_bootstrap"],
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief (1-2 sentence) reasoning for this patch.",
                },
            },
        },
    },
]


class LLMWorker:
    """Owns one thread, one Anthropic session, one queue → one agent's knobs."""

    def __init__(
        self,
        client: Any,
        agent_id: int,
        state: CvCAgentState,
        recorder: EventRecorder | None = None,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._state = state
        self._recorder = recorder if recorder is not None else EventRecorder()
        self._shutdown = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"cvc-llm-a{agent_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._shutdown.set()
        # Wake a blocked read_recent_logs by pushing a sentinel.
        try:
            self._state.log_queue.put_nowait({"__shutdown__": True})
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    # ── tool implementations ────────────────────────────────────────────

    def _tool_read_recent_logs(self, args: dict) -> dict:
        max_events = int(args.get("max_events", 20))
        events: list[dict] = []
        try:
            first = self._state.log_queue.get(timeout=_READ_TIMEOUT_S)
        except queue.Empty:
            return {"events": []}
        if first.get("__shutdown__"):
            return {"shutdown": True, "events": []}
        events.append(first)
        while len(events) < max_events:
            try:
                e = self._state.log_queue.get_nowait()
            except queue.Empty:
                break
            if e.get("__shutdown__"):
                return {"shutdown": True, "events": events}
            events.append(e)
        return {"events": events}

    def _tool_patch(self, args: dict) -> dict:
        applied: dict[str, Any] = {}
        state = self._state
        if args.get("resource_bias"):
            state.resource_bias_from_llm = args["resource_bias"]
            applied["resource_bias"] = args["resource_bias"]
        if args.get("role"):
            state.llm_role_override = args["role"]
            applied["role"] = args["role"]
        if args.get("objective"):
            state.llm_objective = args["objective"]
            applied["objective"] = args["objective"]
        rationale = args.get("rationale", "")
        state.llm_log.append(
            {
                "agent": self._agent_id,
                "type": "patch",
                "applied": applied,
                "rationale": rationale,
            }
        )
        self._recorder.emit(
            type="patch_applied",
            agent=self._agent_id,
            stream="llm",
            payload={"applied": applied, "rationale": rationale},
        )
        return {"ok": True, "applied": applied}

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        t0 = time.perf_counter()
        if name == "read_recent_logs":
            out = self._tool_read_recent_logs(args)
        elif name == "patch":
            out = self._tool_patch(args)
        else:
            out = {"error": f"unknown tool: {name}"}
        latency_ms = (time.perf_counter() - t0) * 1000
        self._recorder.emit(
            type="llm_tool_call",
            agent=self._agent_id,
            stream="llm",
            payload={
                "tool": name,
                "input": dict(args),
                "latency_ms": round(latency_ms, 2),
            },
        )
        return out

    # ── main loop ───────────────────────────────────────────────────────

    def _initial_messages(self) -> list[dict]:
        return [
            {
                "role": "user",
                "content": (
                    f"You are coaching agent {self._agent_id} for this episode. "
                    "Start by calling read_recent_logs. Continue looping "
                    "(read_recent_logs → reason → patch when needed) until the "
                    "episode ends."
                ),
            }
        ]

    def _step_once(self, messages: list[dict] | None = None) -> bool:
        """Run one request/response round-trip. Returns True if the loop
        should exit (shutdown sentinel or end_turn in single-step tests)."""
        if messages is None:
            if not hasattr(self, "_messages"):
                self._messages = self._initial_messages()
            messages = self._messages

        t0 = time.perf_counter()
        response = self._client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            system=_SYSTEM,
            tools=_TOOLS,
            messages=messages,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        self._state.llm_latencies.append(latency_ms)

        messages.append({"role": "assistant", "content": response.content})

        stop = False
        if response.stop_reason == "tool_use":
            tool_results: list[dict] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                out = self._dispatch_tool(block.name, dict(block.input or {}))
                if out.get("shutdown"):
                    stop = True
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(out),
                    }
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            # No tool call. Nudge the model to resume polling.
            messages.append(
                {"role": "user", "content": "Continue. Call read_recent_logs."}
            )
            stop = True

        if len(messages) > _HISTORY_TRIM_AT:
            messages[:] = [messages[0]] + messages[-_HISTORY_KEEP_TAIL:]
        return stop

    def _run(self) -> None:
        self._messages = self._initial_messages()
        while not self._shutdown.is_set():
            # _step_once returns True on end_turn or shutdown. In production
            # we keep looping regardless of end_turn (a nudge was appended);
            # we only exit when the shutdown event is set.
            self._step_once(self._messages)
