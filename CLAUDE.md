# CvC Policy

Minimal LLM+Python policy for CoGames submission. Skills and tooling live in the cogamer repo.

## Commands

```bash
pip install -e ".[llm]"                    # Install with LLM support
pytest tests/ -v                           # Run all tests

# Play
softmax cogames play -m machina_1 -p class=cvc_policy.cogamer_policy.CvCPolicy --render=gui

# Evaluate
softmax cogames eval -m machina_1 -p class=cvc_policy.cogamer_policy.CvCPolicy -e 10 --format json

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

## Non-Negotiables

1. Let it crash — no try/except for error hiding
2. Minimal diffs — smallest change that fixes the root cause
3. Fix root causes, not symptoms
