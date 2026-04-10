# CvC Policy

## What This Is

A CvC (Claude vs Claude) policy for CoGames. The goal is to improve this policy's performance and submit it to compete.

## Commands

```bash
# Play a game (visual)
cogames play -m machina_1 -p class=cvc_policy.cogamer_policy.CvCPolicy --render=gui

# Play headless with trace
cogames play -m machina_1 -p class=cvc_policy.cogamer_policy.CvCPolicy --render=log

# Evaluate
cogames evaluate -m machina_1 -p class=cvc_policy.cogamer_policy.CvCPolicy -e 10 --format json

# Submit
cogames upload -p class=cvc_policy.cogamer_policy.CvCPolicy -n <name> --setup-script setup_policy.py
```

## Key Files

- `src/cvc_policy/cogamer_policy.py` — CvCPolicy entry point, LLM↔Python bridge
- `src/cvc_policy/programs.py` — program table (32 programs, main evolvable surface)
- `src/cvc_policy/game_state.py` — observation processing, state management
- `src/cvc_policy/agent/main.py` — CvcEngine decision tree
- `src/cvc_policy/agent/roles.py` — role-specific actions (miner, aligner, scrambler)
- `src/cvc_policy/agent/targeting.py` — target selection and scoring
- `src/cvc_policy/agent/pressure.py` — role budgets and retreat thresholds
- `docs/architecture.md` — architecture reference with alpha.0 comparison

## Trace

The policy writes LLM↔Python communication traces to `/tmp/cvc-trace/` (configurable via `CVC_TRACE_DIR`). Each trace entry includes the prompt sent to the LLM, raw response, parsed fields, and latency.

## Testing

```bash
pytest tests/ -v
```

## Non-Negotiables

1. Let it crash — no try/except for error hiding
2. Minimal diffs — smallest change that fixes the root cause
3. Fix root causes, not symptoms
