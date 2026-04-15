"""Static HTML report generation for a single run folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from cvc_policy.recorder import payload_text


# 8 distinguishable hex colors for agent tags (agent_id % 8).
_AGENT_PALETTE: list[str] = [
    "#10b981",  # emerald
    "#3b82f6",  # blue
    "#a855f7",  # purple
    "#f59e0b",  # amber
    "#0ea5e9",  # sky
    "#f43f5e",  # rose
    "#84cc16",  # lime
    "#d946ef",  # fuchsia
]


def agent_color(agent_id: int) -> str:
    """Stable per-agent foreground color keyed by `agent_id % 8`."""
    return _AGENT_PALETTE[int(agent_id) % len(_AGENT_PALETTE)]


_ROLE_GLYPHS: dict[str, str] = {
    "miner": "\u26cf",          # pickaxe
    "aligner": "\U0001f517",    # link
    "scrambler": "\U0001f300",  # cyclone / spiral
    "scout": "\U0001f52d",      # telescope
}


def role_glyph(role: Any) -> str:
    """Return the unicode glyph for a role, or empty string for unknown/None."""
    if not isinstance(role, str):
        return ""
    return _ROLE_GLYPHS.get(role, "")


def _safe_script_json(obj: Any) -> str:
    """Serialize to JSON and neutralize sequences that would break out
    of a `<script type="application/json">` island (XSS vector).

    Also escapes any remaining bare `<` as `\\u003c` so arbitrary HTML
    tags inside JSON strings cannot appear verbatim in the rendered
    source. All replacements round-trip through `JSON.parse`.
    """
    s = json.dumps(obj, ensure_ascii=False)
    _PLACEHOLDER = "\x00CLOSESLASH\x00"
    return (
        s.replace("</", _PLACEHOLDER)
        .replace("<!--", "\\u003c!\\-\\-")
        .replace("<![CDATA[", "\\u003c![CDATA\\[")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
        .replace("<", "\\u003c")
        .replace(_PLACEHOLDER, "<\\/")
    )

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
        autoescape=True,
    )


def _type_counts(events: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    out = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(t, n, TYPE_COLORS.get(t, _DEFAULT_COLOR)) for t, n in out]


def _group_by_step(
    events: list[dict[str, Any]], max_step: int
) -> list[dict[str, Any]]:
    """Group events into display groups for the log panel.

    Each group is one of:
      - {"type": "step", "step": N, "events": [...]} for a step that has
        at least one event, or for a single empty step sandwiched between
        populated steps.
      - {"type": "range", "start": A, "end": B} for a contiguous run of
        **two or more** consecutive empty steps, A < B.

    Groups cover every step in `0..max_step` inclusive, so a run with no
    events produces exactly one `range` group `[0, max_step]`.
    """
    by_step: dict[int, list[dict[str, Any]]] = {}
    for e in events:
        s = int(e.get("step", 0))
        by_step.setdefault(s, []).append(e)

    groups: list[dict[str, Any]] = []
    i = 0
    while i <= max_step:
        if i in by_step:
            groups.append({"type": "step", "step": i, "events": by_step[i]})
            i += 1
            continue
        # Empty-step run starts at i; walk forward while empty.
        j = i
        while j <= max_step and j not in by_step:
            j += 1
        run_len = j - i  # number of empty steps
        if run_len >= 2:
            groups.append({"type": "range", "start": i, "end": j - 1})
        else:
            # Single empty step — keep it as a bare step marker.
            groups.append({"type": "step", "step": i, "events": []})
        i = j
    return groups


def _merge_duplicate_steps(
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse adjacent ``step`` groups that each hold exactly one
    equivalent event into a single ``step`` group with ``step_end`` set.

    Equivalence key: ``(agent, stream, type, payload_text(event))`` —
    rendered payload text, not the raw dict, so volatile fields formatted
    by the payload renderer compare structurally.

    ``range`` groups (empty-step runs) block merging: the two sides of a
    range are never fused.
    """
    def _key(e: dict[str, Any]) -> tuple[Any, Any, Any, str]:
        return (e.get("agent"), e.get("stream"), e["type"], payload_text(e))

    out: list[dict[str, Any]] = []
    i = 0
    n = len(groups)
    while i < n:
        g = groups[i]
        if (
            g["type"] == "step"
            and len(g["events"]) == 1
        ):
            start = g["step"]
            end = start
            k = _key(g["events"][0])
            j = i + 1
            while j < n:
                h = groups[j]
                if h["type"] != "step":
                    break
                if len(h["events"]) != 1:
                    break
                if h["step"] != end + 1:
                    break
                if _key(h["events"][0]) != k:
                    break
                end = h["step"]
                j += 1
            merged = {
                "type": "step",
                "step": start,
                "step_end": end,
                "events": [g["events"][0]],
            }
            out.append(merged)
            i = j
            continue
        out.append(g)
        i += 1
    return out


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

    # Track the most recent action-emitted role per agent so non-action
    # events inherit the icon of that agent's last role.
    last_role_by_agent: dict[int, str] = {}
    role_by_event_idx: dict[int, str | None] = {}
    for i, e in enumerate(events):
        agent = e.get("agent")
        if agent is None:
            role_by_event_idx[i] = None
            continue
        payload = e.get("payload") or {}
        if e.get("type") == "action":
            r = payload.get("role")
            if isinstance(r, str):
                last_role_by_agent[int(agent)] = r
        role_by_event_idx[i] = last_role_by_agent.get(int(agent))

    # Pre-render log lines for the right-side panel as structured dicts.
    # The template composes the final line with styled spans for stream
    # and agent — `text` is just the payload portion (no [stream]/a<N>
    # prefix).
    def _as_line(idx: int, e: dict[str, Any]) -> dict[str, Any]:
        return {
            "idx": idx,
            "step": int(e.get("step", 0)),
            "agent": e.get("agent"),
            "stream": e.get("stream", ""),
            "type": e["type"],
            "text": payload_text(e),
            "role": role_by_event_idx.get(idx),
        }

    # Stable event-to-index map so `data-idx` matches the embedded
    # events JSON across all downstream consumers (scrubber, filters).
    # Duplicate-merging is deferred to client-side JS so it can recompose
    # correctly as filters toggle. `_merge_duplicate_steps` remains a pure
    # tested helper but is no longer called here.
    idx_of = {id(e): i for i, e in enumerate(events)}
    raw_groups = _group_by_step(events, max_step)
    log_groups: list[dict[str, Any]] = []
    for g in raw_groups:
        if g["type"] == "range":
            log_groups.append(g)
            continue
        lines: list[dict[str, Any]] = []
        prev_agent_in_step: Any = object()  # sentinel distinct from None
        for e in g["events"]:
            ln = _as_line(idx_of[id(e)], e)
            # Hide the agent+role glyphs when the preceding line in this
            # step block has the same agent — the agent column is the
            # visual "thread" that the follow-up events belong to.
            ln["hide_agent"] = (
                ln["agent"] is not None and ln["agent"] == prev_agent_in_step
            )
            lines.append(ln)
            prev_agent_in_step = ln["agent"]
        log_groups.append({
            "type": "step",
            "step": g["step"],
            "step_end": None,
            "lines": lines,
        })

    # Pre-compute inventory snapshots per agent for the inventory panel.
    # Any event whose payload carries `inventory` or `hp` contributes
    # (action events fire every tick; heartbeats add role/team_resources).
    inventory_by_agent_step: dict[str, list[dict[str, Any]]] = {}
    # Map each agent to its most recently observed team id, and collect
    # team-level snapshots (team_resources + junctions) indexed by team.
    agent_to_team: dict[int, str] = {}
    team_by_step: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        payload = e.get("payload") or {}
        if not ("inventory" in payload or "hp" in payload):
            continue
        a = e.get("agent")
        if a is None:
            continue
        key = str(int(a))
        inventory_by_agent_step.setdefault(key, []).append({
            "step": int(e.get("step", 0)),
            "payload": dict(payload),
        })
        team = payload.get("team")
        if isinstance(team, str) and team:
            agent_to_team[int(a)] = team
            team_payload: dict[str, Any] = {}
            if "team_resources" in payload:
                team_payload["team_resources"] = payload["team_resources"]
            if "junctions" in payload:
                team_payload["junctions"] = payload["junctions"]
            if team_payload:
                team_by_step.setdefault(team, []).append({
                    "step": int(e.get("step", 0)),
                    "payload": team_payload,
                })
    for entries in inventory_by_agent_step.values():
        entries.sort(key=lambda r: r["step"])
    for entries in team_by_step.values():
        entries.sort(key=lambda r: r["step"])

    # Group agents by team id, with agents lacking a team id falling under
    # the empty-string bucket. Teams sorted by team id; agents keep their
    # existing order within each team.
    team_groups: list[dict[str, Any]] = []
    by_team: dict[str, list[int]] = {}
    for a in agents:
        by_team.setdefault(agent_to_team.get(a, ""), []).append(a)
    for team in sorted(by_team):
        team_groups.append({"team": team, "agents": by_team[team]})

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
        "type_counts": _type_counts(events),
        "log_groups": log_groups,
        "events_json": _safe_script_json(events),
        "inventory_by_agent_step_json": _safe_script_json(
            inventory_by_agent_step
        ),
        "team_by_step_json": _safe_script_json(team_by_step),
        "team_groups": team_groups,
        "failed": failed,
        "assertions": result.get("assertions", []),
        "has_replay": has_replay,
        "replay_rel": str(run_dir.resolve() / "replay.json.z"),
        "replay_abs_path": str(run_dir.resolve() / "replay.json.z"),
    }

    env = _env()
    env.globals["agent_color"] = agent_color
    env.globals["role_glyph"] = role_glyph
    html = env.get_template("report.html.j2").render(**ctx)
    out = run_dir / "report.html"
    out.write_text(html)
    return out
