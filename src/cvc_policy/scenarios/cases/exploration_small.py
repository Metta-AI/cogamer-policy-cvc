"""S1 exploration_small — tier 1.

Tutorial mission with miner variant, 1 cog, 300 steps. Asserts the
agent accumulated meaningful world-model state during the run.

We use `known_cells_at_least` rather than `map_coverage_at_least`
because WorldModel does not track free cells — the summary payload's
known/reachable ratio is a placeholder. The raw entity count is a
direct signal that observation processing ran and the agent moved
around. Calibrated against seed=42 on 2026-04-13: run produced
known_cells=112; we require >= 30 as a conservative lower bound.
Extractors are pruned from the world model when not visible, so we
cannot assert extractors_known>0 at end of run reliably.

See design doc §7a for the placeholder caveat.
"""

from __future__ import annotations

from cvc_policy.scenarios import Scenario, scenario
from cvc_policy.scenarios.assertions import known_cells_at_least, no_crash


@scenario
def exploration_small() -> Scenario:
    return Scenario(
        name="exploration_small",
        tier=1,
        mission="tutorial.miner",
        cogs=1,
        steps=300,
        seed=42,
        assertions=[
            no_crash(),
            known_cells_at_least(agent=0, minimum=30),
        ],
    )
