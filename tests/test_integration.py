"""Integration tests that require an LLM API key.

Run with: pytest tests/test_integration.py -v
Skipped automatically when ANTHROPIC_API_KEY is not set.
"""

from __future__ import annotations

import os
import json
import subprocess
import glob

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


class TestLLMParsePipeline:
    """Verify the full LLM prompt -> response -> parse -> apply pipeline."""

    def test_llm_response_parses_successfully(self):
        """Send the actual prompt to the LLM and confirm _parse_analysis extracts fields."""
        from cvc_policy.programs import _build_analysis_prompt, _parse_analysis

        import anthropic

        ctx = {
            "step": 500,
            "agent_id": "0",
            "hp": 100,
            "inventory": {"heart": 2, "carbon": 5},
            "team_resources": {"carbon": 200, "oxygen": 150, "germanium": 50, "silicon": 100},
            "has_gear": True,
            "roles": "miner=3, aligner=3, scrambler=2",
            "position": (-10, 15),
            "junctions": {"friendly": 5, "enemy": 3, "neutral": 2},
            "stalled": False,
            "oscillating": False,
            "safe_distance": 20,
            "role": "miner",
        }
        prompt = _build_analysis_prompt(ctx)

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text
        parsed = _parse_analysis(text)

        assert "resource_bias" in parsed, f"Failed to parse resource_bias from: {text}"
        assert parsed["resource_bias"] in ("carbon", "oxygen", "germanium", "silicon")
        assert "analysis" in parsed
        assert len(parsed["analysis"]) > 0


class TestPlayIntegration:
    """Run a short game and verify trace output."""

    def test_short_game_produces_trace(self, tmp_path):
        """Run 600 steps (enough for one LLM call at step 500) and check trace."""
        trace_dir = str(tmp_path / "trace")
        os.makedirs(trace_dir, exist_ok=True)

        env = os.environ.copy()
        env["CVC_TRACE_DIR"] = trace_dir

        result = subprocess.run(
            [
                "softmax", "cogames", "play",
                "-m", "machina_1",
                "-p", "class=cvc_policy.cogamer_policy.CvCPolicy",
                "--render=log",
                "-s", "600",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=env,
        )
        assert result.returncode == 0, f"Game failed: {result.stderr[-500:]}"

        trace_files = glob.glob(os.path.join(trace_dir, "*.json"))
        assert len(trace_files) > 0, "No trace files produced"

        trace = json.loads(open(trace_files[0]).read())
        assert "agents" in trace
        assert "llm_trace" in trace
        assert len(trace["agents"]) == 8

        # At 600 steps, LLM should have fired at step 500
        assert len(trace["llm_trace"]) > 0, "No LLM calls recorded in trace"

        # Check at least some agents got resource_bias applied
        agents_with_bias = [
            a for a in trace["agents"].values()
            if a["final_resource_bias"] is not None
        ]
        assert len(agents_with_bias) > 0, (
            "No agents received resource_bias from LLM — parse pipeline is broken"
        )
