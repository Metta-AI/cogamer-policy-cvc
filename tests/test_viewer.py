"""Tests for the HTML report viewer (Batch 3)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


def _write_fake_run(
    run_dir: Path,
    *,
    run_id: str = "fake-run-20260101-000000",
    scenario: str = "fake_scenario",
    status: str = "passed",
    cogs: int = 2,
    steps: int = 10,
    seed: int = 42,
    duration_s: float = 1.5,
    events: list[dict] | None = None,
    assertions: list[dict] | None = None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    events = events if events is not None else [
        {"step": 0, "agent": 0, "stream": "py", "type": "action",
         "payload": {"role": "miner", "summary": "noop"}},
        {"step": 1, "agent": 0, "stream": "py", "type": "role_change",
         "payload": {"from": "miner", "to": "aligner"}},
        {"step": 2, "agent": 1, "stream": "py", "type": "target",
         "payload": {"kind": "carbon_extractor", "pos": [5, 5], "distance": 3}},
        {"step": 3, "agent": None, "stream": "py", "type": "note",
         "payload": {"text": "hello"}},
        {"step": 5, "agent": 0, "stream": "llm", "type": "llm_tool_call",
         "payload": {"tool": "patch", "input": {}, "latency_ms": 100}},
    ]
    (run_dir / "events.json").write_text(json.dumps(events))
    result = {
        "run_id": run_id,
        "scenario": scenario,
        "started_at": "2026-01-01T00:00:00",
        "duration_s": duration_s,
        "steps": steps,
        "cogs": cogs,
        "mission": "tutorial.miner",
        "variants": [],
        "seed": seed,
        "policy_kwargs": {},
        "status": status,
        "assertions": assertions if assertions is not None else [
            {"name": "no_crash", "passed": True, "message": "ok",
             "failed_at_step": None},
        ],
    }
    (run_dir / "result.json").write_text(json.dumps(result))
    return run_dir


def test_render_writes_report_html_non_empty(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(tmp_path / "fake-run")
    out = render(run_dir)
    assert out == run_dir / "report.html"
    assert out.exists()
    assert out.stat().st_size > 0


def test_report_contains_run_id_and_scenario(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(
        tmp_path / "r", run_id="my-run-id-123", scenario="my_scenario"
    )
    html = render(run_dir).read_text()
    assert "my-run-id-123" in html
    assert "my_scenario" in html


def test_report_has_one_svg_per_agent(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(tmp_path / "r", cogs=3)
    html = render(run_dir).read_text()
    # One timeline SVG per agent.
    svgs = re.findall(r"<svg[^>]*class=\"timeline\"", html)
    assert len(svgs) == 3


def test_report_embeds_events_json_blob(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    events = [
        {"step": 0, "agent": 0, "stream": "py", "type": "action",
         "payload": {"role": "miner"}},
        {"step": 3, "agent": None, "stream": "py", "type": "note",
         "payload": {"text": "team event"}},
    ]
    run_dir = _write_fake_run(tmp_path / "r", events=events)
    html = render(run_dir).read_text()
    m = re.search(
        r'<script type="application/json" id="events">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert m is not None
    parsed = json.loads(m.group(1))
    assert parsed == events


def test_report_failure_view_shows_failed_assertion(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(
        tmp_path / "r",
        status="failed",
        assertions=[
            {"name": "mining_trips_efficient", "passed": False,
             "message": "plateau waste", "failed_at_step": 42},
        ],
    )
    html = render(run_dir).read_text()
    assert "mining_trips_efficient" in html
    assert "failed" in html.lower()
    assert "42" in html  # failed_at_step surfaced


def test_report_tick_has_data_attrs(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    events = [
        {"step": 7, "agent": 0, "stream": "py", "type": "target",
         "payload": {"kind": "carbon_extractor", "pos": [1, 1]}},
    ]
    run_dir = _write_fake_run(tmp_path / "r", cogs=1, events=events)
    html = render(run_dir).read_text()
    # Tick should carry data-step / data-agent / data-type attributes.
    assert 'data-step="7"' in html
    assert 'data-agent="0"' in html
    assert 'data-type="target"' in html


def test_report_replay_card_mentions_replay_file(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(tmp_path / "my-run")
    (run_dir / "replay.json.z").write_bytes(b"fake")
    html = render(run_dir).read_text()
    assert "replay.json.z" in html
    assert "softmax cogames replay" in html


def test_render_neutralizes_script_end_in_json_island(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(
        tmp_path / "r",
        events=[
            {"step": 0, "agent": 0, "stream": "py", "type": "note",
             "payload": {"text": "oops </script><script>alert(1)</script>"}},
        ],
    )
    html = render(run_dir).read_text()
    # Extract only the JSON island and assert no `</script>` breaks out.
    m = re.search(
        r'<script type="application/json" id="events">(.*?)</script>',
        html,
        re.DOTALL,
    )
    assert m is not None
    island = m.group(1)
    assert "</script>" not in island
    # Escaped form should appear inside the island.
    assert "<\\/script>" in island
    # And the island JSON must still round-trip to the original payload.
    parsed = json.loads(island)
    assert parsed[0]["payload"]["text"] == "oops </script><script>alert(1)</script>"


def test_render_escapes_event_text_in_log_panel(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(
        tmp_path / "r",
        events=[
            {"step": 0, "agent": 0, "stream": "py", "type": "note",
             "payload": {"text": "<img src=x onerror=alert(1)>"}},
        ],
    )
    html = render(run_dir).read_text()
    assert "<img src=x onerror=alert(1)>" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_render_escapes_assertion_message(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(
        tmp_path / "r",
        status="failed",
        assertions=[
            {"name": "bad", "passed": False,
             "message": "<script>alert('x')</script>",
             "failed_at_step": 3},
        ],
    )
    html = render(run_dir).read_text()
    assert "<script>alert('x')</script>" not in html
    assert "&lt;script&gt;alert(" in html


def test_report_replay_kbd_is_absolute_path(tmp_path: Path) -> None:
    from cvc_policy.viewer import render

    run_dir = _write_fake_run(tmp_path / "my-run")
    (run_dir / "replay.json.z").write_bytes(b"fake")
    html = render(run_dir).read_text()
    m = re.search(r'<kbd id="replay-cmd">([^<]+)</kbd>', html)
    assert m is not None
    text = m.group(1)
    assert text.startswith("softmax cogames replay ")
    path_part = text[len("softmax cogames replay "):].strip()
    assert Path(path_part).is_absolute()
    assert Path(path_part).exists()
    assert "replay.json.z" in path_part


def test_cgp_view_rejects_path_traversal(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from cvc_policy.cli import app

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    # Sibling directory the attacker is trying to escape into.
    (tmp_path / "escape").mkdir()

    result = CliRunner().invoke(
        app, ["view", "../escape", "--runs-root", str(runs_root)]
    )
    assert result.exit_code != 0
    combined = (result.output or "") + (
        str(result.exception) if result.exception else ""
    )
    assert "traversal" in combined.lower() or "outside" in combined.lower()


def test_cgp_view_no_server_opens_file_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    from cvc_policy.cli import app

    runs_root = tmp_path / "runs"
    _write_fake_run(runs_root / "abc-20260101-000000", run_id="abc")

    opened: list[str] = []
    monkeypatch.setattr(
        "webbrowser.open", lambda url: opened.append(url) or True
    )

    result = CliRunner().invoke(
        app,
        [
            "view",
            "abc-20260101-000000",
            "--runs-root",
            str(runs_root),
            "--no-server",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(opened) == 1
    opened_path = opened[0]
    # file:// URL pointing at an existing report.html
    assert opened_path.startswith("file://") or Path(opened_path).is_absolute()
    path_str = opened_path.replace("file://", "")
    assert Path(path_str).exists()
    assert path_str.endswith("report.html")


def test_cgp_view_with_server_starts_http_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import re as _re
    import urllib.request

    from typer.testing import CliRunner

    from cvc_policy.cli import app

    runs_root = tmp_path / "runs"
    _write_fake_run(runs_root / "abc-20260101-000000", run_id="abc")

    opened: list[str] = []
    monkeypatch.setattr(
        "webbrowser.open", lambda url: opened.append(url) or True
    )

    # Force the blocking loop to exit immediately.
    def _fake_serve_forever(self):  # type: ignore[no-untyped-def]
        raise KeyboardInterrupt

    # Patch ThreadingHTTPServer.serve_forever so `cgp view` returns.
    import http.server

    # The CLI command uses _serve_run helper; by patching serve_forever we let
    # the command start the server, fire open, then unwind via KeyboardInterrupt.
    monkeypatch.setattr(
        http.server.ThreadingHTTPServer,
        "serve_forever",
        _fake_serve_forever,
    )

    result = CliRunner().invoke(
        app,
        [
            "view",
            "abc-20260101-000000",
            "--runs-root",
            str(runs_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(opened) == 1
    url = opened[0]
    m = _re.match(r"^http://localhost:(\d+)/report\.html$", url)
    assert m is not None, f"unexpected url: {url}"


def test_serve_run_helper_actually_serves_report(tmp_path: Path) -> None:
    """The _serve_run helper returns a live server that serves report.html."""
    import urllib.request

    from cvc_policy.cli import _serve_run
    from cvc_policy.viewer import render

    runs_root = tmp_path / "runs"
    run_dir = _write_fake_run(runs_root / "abc-20260101-000000", run_id="abc")
    render(run_dir)

    httpd, port = _serve_run(run_dir)
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/report.html", timeout=5
        ) as resp:
            body = resp.read().decode("utf-8")
        assert "abc" in body
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_cgp_runs_lists_most_recent_first(
    tmp_path: Path,
) -> None:
    from typer.testing import CliRunner

    from cvc_policy.cli import app

    runs_root = tmp_path / "runs"
    _write_fake_run(runs_root / "old-20260101-000000", run_id="old",
                    scenario="scen_a", status="passed")
    _write_fake_run(runs_root / "new-20260102-000000", run_id="new",
                    scenario="scen_b", status="failed")
    # Bump mtime so "new" is newer.
    import os
    os.utime(runs_root / "new-20260102-000000", (2_000_000_000, 2_000_000_000))
    os.utime(runs_root / "old-20260101-000000", (1_000_000_000, 1_000_000_000))

    result = CliRunner().invoke(app, ["runs", "--runs-root", str(runs_root)])
    assert result.exit_code == 0, result.output
    # Most recent first: "new" before "old".
    assert result.output.find("new-20260102-000000") < result.output.find(
        "old-20260101-000000"
    )
    assert "scen_a" in result.output
    assert "scen_b" in result.output
    assert "failed" in result.output
