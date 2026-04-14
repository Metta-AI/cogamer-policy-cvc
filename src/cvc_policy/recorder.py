"""Structured event recorder for CvC policy diagnostics.

Single producer of per-tick events. Events fan out to configured sinks
(stderr, events.json, mettagrid policyInfos). Replaces the stderr-only
LogConfig.
"""

from __future__ import annotations

from typing import Any


def _fmt_value(v: Any) -> str:
    s = str(v)
    if any(c.isspace() for c in s):
        return f"'{s}'"
    return s


def _fmt_payload_default(payload: dict[str, Any]) -> str:
    return " ".join(f"{k}={_fmt_value(v)}" for k, v in payload.items())


def _fmt_payload_patch_applied(payload: dict[str, Any]) -> str:
    applied = payload.get("applied") or {}
    parts = [f"{k}={_fmt_value(v)}" for k, v in applied.items()]
    rationale = payload.get("rationale")
    if rationale:
        parts.append(f"rationale={_fmt_value(rationale)}")
    return " ".join(parts)


_PAYLOAD_RENDERERS: dict[str, Any] = {
    "patch_applied": _fmt_payload_patch_applied,
}


def fmt(event: dict[str, Any]) -> str:
    """Render an event as a single-line stderr string."""
    stream = event["stream"]
    step = event["step"]
    agent = event.get("agent")
    etype = event["type"]
    payload = event.get("payload") or {}
    renderer = _PAYLOAD_RENDERERS.get(etype, _fmt_payload_default)
    body = renderer(payload)
    prefix = f"[{stream}]"
    agent_part = f" a{agent}" if agent is not None else ""
    head = f"{prefix}{agent_part} step={step} {etype}"
    if body:
        return f"{head} {body}"
    return head


class EventRecorder:
    def __init__(self) -> None:
        self._step = 0
        self.events: list[dict[str, Any]] = []

    def set_step(self, step: int) -> None:
        self._step = step

    def emit(
        self,
        *,
        type: str,
        agent: int | None,
        stream: str,
        payload: dict[str, Any],
    ) -> None:
        self.events.append(
            {
                "step": self._step,
                "agent": agent,
                "stream": stream,
                "type": type,
                "payload": dict(payload),
            }
        )
