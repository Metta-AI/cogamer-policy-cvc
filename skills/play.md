# Play

Run a CvC game and capture trace data for analysis.

## Command

```bash
cogames play -m <mission> -p class=cvc --render=log --save-replay-file /tmp/cvc-replay.json.z
```

Defaults: mission=machina_1, policy=class=cvc, steps=1000.

## Trace Output

The CvCPolicy writes an LLM-Python communication trace to `/tmp/cvc-trace/`. Each file is a JSON with:
- `agents`: per-agent step count, LLM call count, final resource bias
- `llm_trace`: chronological list of every LLM call with prompt, raw response, parsed fields, latency

Read the trace after play to understand how the LLM and Python code interacted.

## Customization

- `--mission <name>` or `-m` (run `cogames missions` to list)
- `--steps <n>` or `-s`
- `--render gui` for visual mode, `log` for headless
- `--seed <n>` for reproducibility
- `--save-replay-file <path>` for replay data

## After Play

Read `/tmp/cvc-trace/*.json` for the LLM communication trace. Use `/evaluate` for multi-episode scoring or `/analyze` to diagnose issues from this run.
