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

from cvc_policy.logcfg import LogConfig

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
        log: LogConfig | None = None,
    ) -> None:
        self._client = client
        self._agent_id = agent_id
        self._state = state
        self._log = log if log is not None else LogConfig("")
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
        state.llm_log.append(
            {
                "agent": self._agent_id,
                "type": "patch",
                "applied": applied,
                "rationale": args.get("rationale", ""),
            }
        )
        rationale = args.get("rationale", "").strip()
        self._log.log(
            "llm",
            f"a{self._agent_id} patch {applied}"
            + (f" — {rationale[:120]}" if rationale else ""),
        )
        return {"ok": True, "applied": applied}

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        if name == "read_recent_logs":
            out = self._tool_read_recent_logs(args)
            self._log.log(
                "llm", f"a{self._agent_id} read_recent_logs -> {len(out.get('events', []))} events"
            )
            return out
        if name == "patch":
            return self._tool_patch(args)
        return {"error": f"unknown tool: {name}"}

    # ── main loop ───────────────────────────────────────────────────────

    def _run(self) -> None:
        messages: list[dict] = [
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

        while not self._shutdown.is_set():
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

            if response.stop_reason == "tool_use":
                tool_results: list[dict] = []
                shutdown_requested = False
                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue
                    out = self._dispatch_tool(block.name, dict(block.input or {}))
                    if out.get("shutdown"):
                        shutdown_requested = True
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(out),
                        }
                    )
                messages.append({"role": "user", "content": tool_results})
                if shutdown_requested:
                    return
            else:
                # No tool call. Nudge it to keep going.
                messages.append(
                    {"role": "user", "content": "Continue. Call read_recent_logs."}
                )

            # Prevent unbounded growth. Keep the grounding prompt + recent tail.
            if len(messages) > _HISTORY_TRIM_AT:
                messages = [messages[0]] + messages[-_HISTORY_KEEP_TAIL:]
