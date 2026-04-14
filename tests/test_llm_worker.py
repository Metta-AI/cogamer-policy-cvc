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


def test_tool_call_read_recent_logs_emits_event():
    client = FakeAnthropicClient()
    client.queue_tool_use("read_recent_logs", {})
    client.queue_end_turn()
    worker = _run_worker(client)
    tool_events = [e for e in worker._recorder.events if e["type"] == "llm_tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["payload"]["tool"] == "read_recent_logs"
    assert tool_events[0]["stream"] == "llm"
    assert tool_events[0]["agent"] == 0
    assert "latency_ms" in tool_events[0]["payload"]


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
