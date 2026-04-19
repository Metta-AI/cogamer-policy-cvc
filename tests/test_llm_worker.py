"""Tests for LLMWorker recorder emissions, using a fake Anthropic client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from cvc_policy.cogamer_policy import CvCAgentState
from cvc_policy.llm_worker import LLMWorker
from cvc_policy.recorder import EventRecorder


class _ToolUseBlock(SimpleNamespace):
    type = "tool_use"


class _TextBlock(SimpleNamespace):
    type = "text"


class _Response(SimpleNamespace):
    pass


class FakeAnthropicClient:
    """Scripts a sequence of responses returned by messages.create()."""

    def __init__(self) -> None:
        self._scripted: list[_Response] = []
        self._calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def queue_tool_use(self, tool: str, inp: dict[str, Any], block_id: str = "b1") -> None:
        block = _ToolUseBlock(id=block_id, name=tool, input=dict(inp))
        self._scripted.append(_Response(content=[block], stop_reason="tool_use"))

    def queue_end_turn(self, text: str = "done") -> None:
        block = _TextBlock(text=text)
        self._scripted.append(_Response(content=[block], stop_reason="end_turn"))

    def _create(self, **kwargs: Any) -> _Response:
        self._calls.append(kwargs)
        if not self._scripted:
            # Sentinel end_turn to stop the loop.
            return _Response(
                content=[_TextBlock(text="stop")], stop_reason="end_turn"
            )
        return self._scripted.pop(0)


def _run_worker(client: FakeAnthropicClient, max_iters: int = 10) -> LLMWorker:
    recorder = EventRecorder()
    state = CvCAgentState()
    worker = LLMWorker(client, agent_id=0, state=state, recorder=recorder)
    # Drive the loop body directly (no thread) for determinism.
    from cvc_policy import llm_worker as lw

    orig = lw._READ_TIMEOUT_S
    lw._READ_TIMEOUT_S = 0.01
    try:
        for _ in range(max_iters):
            if worker._step_once():
                break
    finally:
        lw._READ_TIMEOUT_S = orig
    return worker


def test_tool_call_read_recent_logs_emits_llm_turn():
    client = FakeAnthropicClient()
    client.queue_tool_use("read_recent_logs", {})
    client.queue_end_turn()
    worker = _run_worker(client)
    turn_events = [e for e in worker._recorder.events if e["type"] == "llm_turn"]
    assert len(turn_events) >= 1
    first = turn_events[0]
    assert first["stream"] == "llm"
    assert first["agent"] == 0
    assert "latency_ms" in first["payload"]
    assert any(tc["tool"] == "read_recent_logs" for tc in first["payload"]["tool_calls"])


def test_trim_history_never_starts_with_assistant():
    client = FakeAnthropicClient()
    state = CvCAgentState()
    worker = LLMWorker(client, agent_id=0, state=state)
    initial = [{"role": "user", "content": "grounding"}]
    # Build 200 alternating messages: user/assistant/user/assistant/...
    # assistant messages carry tool_use-like blocks so we can exercise the
    # split-pair concern.
    msgs = list(initial)
    for i in range(200):
        if i % 2 == 0:
            msgs.append(
                {
                    "role": "assistant",
                    "content": [_ToolUseBlock(id=f"t{i}", name="x", input={})],
                }
            )
        else:
            msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"t{i - 1}",
                            "content": "ok",
                        }
                    ],
                }
            )
    trimmed = worker._trim_history(msgs)
    # Grounding (msgs[0]) is preserved; msgs[1] — the first message AFTER the
    # grounding in the kept tail — must be a user turn, never an assistant turn.
    assert trimmed[0] is msgs[0]
    assert trimmed[0]["role"] == "user"
    if len(trimmed) > 1:
        assert trimmed[1]["role"] == "user"


def test_read_recent_logs_returns_queued_events():
    """Exercise the queue-draining path with a real log_queue."""
    client = FakeAnthropicClient()
    state = CvCAgentState()
    state.log_queue.put({"type": "action", "step": 1})
    state.log_queue.put({"type": "action", "step": 2})
    worker = LLMWorker(client, agent_id=0, state=state)
    out = worker._tool_read_recent_logs({"max_events": 10})
    assert len(out["events"]) == 2
    assert out["events"][0]["step"] == 1


def test_read_recent_logs_empty_queue():
    client = FakeAnthropicClient()
    state = CvCAgentState()
    worker = LLMWorker(client, agent_id=0, state=state)
    from cvc_policy import llm_worker as lw
    orig = lw._READ_TIMEOUT_S
    lw._READ_TIMEOUT_S = 0.01
    try:
        out = worker._tool_read_recent_logs({})
    finally:
        lw._READ_TIMEOUT_S = orig
    assert out == {"events": []}


def test_read_recent_logs_shutdown_first():
    client = FakeAnthropicClient()
    state = CvCAgentState()
    state.log_queue.put({"__shutdown__": True})
    worker = LLMWorker(client, agent_id=0, state=state)
    out = worker._tool_read_recent_logs({})
    assert out == {"shutdown": True, "events": []}


def test_read_recent_logs_shutdown_mid_batch():
    client = FakeAnthropicClient()
    state = CvCAgentState()
    state.log_queue.put({"type": "action"})
    state.log_queue.put({"__shutdown__": True})
    worker = LLMWorker(client, agent_id=0, state=state)
    out = worker._tool_read_recent_logs({"max_events": 10})
    assert out["shutdown"] is True
    assert len(out["events"]) == 1


def test_patch_tool_role_and_objective():
    client = FakeAnthropicClient()
    state = CvCAgentState()
    worker = LLMWorker(client, agent_id=0, state=state)
    out = worker._tool_patch(
        {"role": "scrambler", "objective": "expand", "rationale": "push"}
    )
    assert out["ok"] is True
    assert out["applied"]["role"] == "scrambler"
    assert out["applied"]["objective"] == "expand"
    assert state.llm_role_override == "scrambler"
    assert state.llm_objective == "expand"


def test_dispatch_unknown_tool():
    client = FakeAnthropicClient()
    state = CvCAgentState()
    worker = LLMWorker(client, agent_id=0, state=state)
    out = worker._dispatch_tool("nope", {})
    assert "error" in out


def test_step_once_handles_shutdown_via_tool():
    client = FakeAnthropicClient()
    # Tool_use triggers read_recent_logs; queue has shutdown sentinel already.
    client.queue_tool_use("read_recent_logs", {})
    state = CvCAgentState()
    state.log_queue.put({"__shutdown__": True})
    worker = LLMWorker(client, agent_id=0, state=state)
    # single step, shutdown=True comes back in tool result; loop continues in
    # production, but _step_once should still record that shutdown was seen
    worker._step_once()


def test_stop_joins_thread_with_shutdown_signal():
    client = FakeAnthropicClient()
    client.queue_end_turn()
    state = CvCAgentState()
    worker = LLMWorker(client, agent_id=0, state=state)
    worker.start()
    worker.stop(timeout=2.0)
    assert worker._shutdown.is_set()


def test_trim_history_short_messages_unchanged():
    client = FakeAnthropicClient()
    state = CvCAgentState()
    worker = LLMWorker(client, agent_id=0, state=state)
    msgs = [{"role": "user", "content": "g"}, {"role": "assistant", "content": "a"}]
    out = worker._trim_history(msgs)
    assert out == msgs


def test_patch_tool_emits_patch_applied_event():
    client = FakeAnthropicClient()
    client.queue_tool_use(
        "patch",
        {"resource_bias": "carbon", "rationale": "low carbon supply"},
    )
    client.queue_end_turn()
    worker = _run_worker(client)
    patch_events = [e for e in worker._recorder.events if e["type"] == "patch_applied"]
    assert len(patch_events) == 1
    assert patch_events[0]["payload"]["applied"] == {"resource_bias": "carbon"}
    assert patch_events[0]["payload"]["rationale"] == "low carbon supply"
    assert patch_events[0]["stream"] == "llm"
