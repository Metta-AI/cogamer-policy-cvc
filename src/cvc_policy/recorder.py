"""Structured event recorder for CvC policy diagnostics.

Single producer of per-tick events. Events fan out to configured sinks
(stderr, events.json, mettagrid policyInfos). Replaces the stderr-only
LogConfig.
"""

from __future__ import annotations

from typing import Any


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
