# Evaluate

Run multi-episode evaluation and produce structured metrics.

## Command

```bash
cogames evaluate -m <mission> -p class=cvc -e <episodes> --format json
```

Defaults: mission=machina_1, episodes=10.

## Reading Results

The JSON output contains per-episode rewards, assignments, and timeouts. Parse it to compute:
- **Average reward** per policy
- **Win rate** across episodes
- **Timeout count** (indicates policy is too slow)

## Multi-Seed Evaluation

For robust comparison (e.g. before/after a code change), run multiple seeds:

```bash
for seed in 42 43 44 45 46; do
  cogames evaluate -m machina_1 -p class=cvc -e 5 --seed $seed --format json
done
```

## Comparing Policies

Evaluate two policies side by side:

```bash
cogames evaluate -m machina_1 -p class=cvc -p class=baseline -e 10 --format json
```

## Customization

- `--mission <name>` or `-m` (supports wildcards)
- `--episodes <n>` or `-e`
- `--seed <n>` for reproducibility
- `--steps <n>` or `-s` to override max steps
- `--format json` or `--format yaml`

## After Evaluation

Use the metrics to establish baselines or confirm improvements. Feed results to `/analyze` for diagnosis.
