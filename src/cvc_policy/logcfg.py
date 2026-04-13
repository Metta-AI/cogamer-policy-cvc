"""Shared log stream config for the policy.

Parsed from the `log` policy kwarg (or CVC_LOG env var). Value is a
`+`-separated list of streams; known streams are `py` (Python-side
decisions and per-tick events) and `llm` (LLM worker activity —
prompts, tool calls, patch applications). `all` is shorthand for
both. Unknown streams are silently ignored so older CLI configs
keep working as streams are added.
"""

from __future__ import annotations

import os
import sys


_KNOWN = frozenset({"py", "llm"})


class LogConfig:
    __slots__ = ("_streams",)

    def __init__(self, spec: str | None) -> None:
        if spec is None:
            spec = os.environ.get("CVC_LOG", "")
        parts = {s.strip().lower() for s in spec.split("+") if s.strip()}
        if "all" in parts:
            parts = set(_KNOWN)
        self._streams: frozenset[str] = frozenset(parts & _KNOWN)

    def enabled(self, stream: str) -> bool:
        return stream in self._streams

    def log(self, stream: str, msg: str) -> None:
        if stream in self._streams:
            print(f"[{stream}] {msg}", flush=True, file=sys.stderr)

    def __repr__(self) -> str:
        return f"LogConfig(streams={sorted(self._streams)})"
