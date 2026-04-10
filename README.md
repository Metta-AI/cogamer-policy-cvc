# CvC Policy

A CvC (Claude vs Claude) policy for [CoGames](https://github.com/Metta-AI/cogames). Clone this repo, improve the policy, and submit to compete.

## Quick Start

```bash
# Clone
git clone https://github.com/Metta-AI/cogamer-policy-cvc.git
cd cogamer-policy-cvc

# Install
pip install -e ".[llm]"

# Play a game
cogames play -m machina_1 -p class=cvc_policy.cogamer_policy.CvCPolicy --render=gui

# Evaluate
cogames evaluate -m machina_1 -p class=cvc_policy.cogamer_policy.CvCPolicy -e 10 --format json

# Submit
cogames upload -p class=cvc_policy.cogamer_policy.CvCPolicy -n my-policy --setup-script setup_policy.py
```

## Improvement Loop

Use the skills in `skills/` to iterate:

1. **Play** (`skills/play.md`) — run a game, capture LLM↔Python trace
2. **Evaluate** (`skills/evaluate.md`) — multi-episode scoring
3. **Analyze** (`skills/analyze.md`) — diagnose the biggest weakness
4. **Improve** (`skills/improve.md`) — implement one fix, verify, submit

## Architecture

The policy is a program table with 32 programs operating on `GameState`:

- **Query programs** — read game state (HP, position, inventory, junctions)
- **Action programs** — movement via A* pathfinding
- **Decision programs** — compose queries + actions (role selection, mining, combat)
- **LLM program** — periodic Claude calls for strategic analysis

See `docs/architecture.md` for details.

## Structure

```
src/cvc_policy/          # Policy implementation
  cogamer_policy.py      # CvCPolicy entry point
  programs.py            # Program table (32 programs)
  game_state.py          # Observation processing + state
  agent/                 # Engine: roles, targeting, navigation, etc.
docs/                    # Architecture and strategy reference
skills/                  # Claude Code improvement skills
tests/                   # Unit tests
setup_policy.py          # Setup script for cogames upload
```
