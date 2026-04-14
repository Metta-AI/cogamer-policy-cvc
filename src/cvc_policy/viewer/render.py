"""Static HTML report generation for a single run folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from cvc_policy.recorder import fmt

TYPE_COLORS: dict[str, str] = {
    "action": "#cbd5e1",
    "role_change": "#f59e0b",
    "target": "#3b82f6",
    "cap_discovered": "#10b981",
    "heartbeat": "#64748b",
    "llm_tool_call": "#a855f7",
    "patch_applied": "#ef4444",
    "note": "#6b7280",
    "error": "#b91c1c",
    "world_model_summary": "#0ea5e9",
}
_DEFAULT_COLOR = "#9ca3af"


def _env() -> Environment:
    tpl_dir = Path(__file__).parent
    return Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        autoescape=select_autoescape(["html"]),
    )


def _type_counts(events: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    out = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(t, n, TYPE_COLORS.get(t, _DEFAULT_COLOR)) for t, n in out]


def _agent_ids(events: list[dict[str, Any]], cogs: int) -> list[int]:
    seen = {e["agent"] for e in events if e.get("agent") is not None}
    ids = sorted(int(a) for a in seen)
    # Pad to cogs so every agent row appears even if it never emitted.
    for i in range(cogs):
        if i not in ids:
            ids.append(i)
    return sorted(set(ids))[:max(cogs, len(ids))]


def render(run_dir: Path) -> Path:
    """Render `report.html` inside `run_dir` and return its path."""
    run_dir = Path(run_dir)
    events_path = run_dir / "events.json"
    result_path = run_dir / "result.json"
    events: list[dict[str, Any]] = (
        json.loads(events_path.read_text()) if events_path.exists() else []
    )
    result: dict[str, Any] = (
        json.loads(result_path.read_text()) if result_path.exists() else {}
    )

    cogs = int(result.get("cogs", 1) or 1)
    agents = _agent_ids(events, cogs)
    max_step = max((int(e.get("step", 0)) for e in events), default=0)

    # Per-agent timeline rows: list of {agent, ticks: [{step, type, color, x_pct}]}
    agent_rows: list[dict[str, Any]] = []
    for a in agents:
        ticks = []
        for e in events:
            if e.get("agent") != a:
                continue
            step = int(e.get("step", 0))
            x_pct = (step / max_step * 100.0) if max_step > 0 else 0.0
            ticks.append({
                "step": step,
                "type": e["type"],
                "color": TYPE_COLORS.get(e["type"], _DEFAULT_COLOR),
                "x_pct": x_pct,
            })
        agent_rows.append({"agent": a, "ticks": ticks})

    # Pre-render log lines for the right-side panel.
    log_lines = []
    for i, e in enumerate(events):
        log_lines.append({
            "idx": i,
            "step": int(e.get("step", 0)),
            "agent": e.get("agent"),
            "stream": e.get("stream", ""),
            "type": e["type"],
            "text": fmt(e),
        })

    failed = [a for a in result.get("assertions", []) if not a.get("passed")]
    status = result.get("status", "unknown")
    has_replay = (run_dir / "replay.json.z").exists()

    ctx = {
        "run_id": result.get("run_id", run_dir.name),
        "scenario": result.get("scenario") or "manual",
        "status": status,
        "duration_s": float(result.get("duration_s") or 0.0),
        "cogs": cogs,
        "seed": result.get("seed"),
        "mission": result.get("mission", ""),
        "variants": result.get("variants", []) or [],
        "steps": result.get("steps", max_step),
        "max_step": max_step,
        "agents": agents,
        "agent_rows": agent_rows,
        "type_counts": _type_counts(events),
        "log_lines": log_lines,
        "events_json": json.dumps(events),
        "failed": failed,
        "assertions": result.get("assertions", []),
        "has_replay": has_replay,
        "replay_rel": f"runs/{run_dir.name}/replay.json.z",
    }

    env = _env()
    html = env.get_template("report.html.j2").render(**ctx)
    out = run_dir / "report.html"
    out.write_text(html)
    return out
