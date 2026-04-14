"""world_model_summary event emission at episode end."""

from __future__ import annotations

from cvc_policy.agent.world_model import WorldModel


def test_empty_world_model_summary() -> None:
    wm = WorldModel()
    s = wm.summary()
    assert s["known_cells"] == 0
    assert s["frontier_cells"] == 0
    assert s["extractors_known"] == 0
    assert s["reachable_cells"] >= 1
