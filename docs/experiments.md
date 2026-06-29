# Running Experiments

This guide covers how to run matches, tournaments, and analyze results. See [`setup.md`](setup.md) for prerequisites.

## Single match (daemon only)

Start the daemon and wait for a game client to connect:

```bash
scripts/run_local_match.sh --p1-policy baseline/random --p2-policy baseline/block_always
```

Or with a provider-backed policy:

```bash
ANTHROPIC_API_KEY=sk-ant-... scripts/run_local_match.sh \
  --p1-policy anthropic/claude --p2-policy baseline/greedy_damage \
  --trace-seed 42
```

The daemon listens on `ws://127.0.0.1:8765` and waits for the mod to connect. Launch YOMI Hustle with the mod enabled to start the match.

When the match ends, artifacts are written to `runs/<timestamp>_<match_id>/`.

## Live match (with mod install check)

For a more complete workflow that checks mod installation:

```bash
scripts/run_live_match.sh --game-dir /path/to/YomiHustle \
  --p1-policy anthropic/claude --p2-policy baseline/random
```

This script:
1. Checks that `uv` is installed
2. Verifies the mod zip exists (run `scripts/package_mod.sh` first)
3. Checks mod installation in the game directory
4. Warns about missing API keys for provider policies
5. Starts the daemon and waits for the game to connect
6. Prints a result summary when the match ends

## Tournament (round-robin)

Run a round-robin tournament across multiple policies:

```bash
scripts/run_round_robin.sh baseline/random baseline/block_always baseline/greedy_damage
```

With a custom config:

```bash
scripts/run_round_robin.sh --config daemon/config/tournament.json \
  baseline/random baseline/block_always baseline/greedy_damage baseline/scripted_safe
```

### Preview pairings without running

```bash
scripts/run_round_robin.sh schedule baseline/random baseline/block_always baseline/greedy_damage
```

This prints the match schedule with side-swap and mirror-match annotations.

### Tournament config options

Key `tournament` fields in the daemon config:

| Field | Default | Description |
|---|---|---|
| `format` | `round_robin` | `single`, `side_swapped_pair`, `round_robin`, `double_round_robin` |
| `games_per_pair` | `10` | Matches per ordered pair |
| `side_swap` | `true` | Alternate sides within each pair |
| `mirror_matches_first` | `true` | Schedule self-play before cross-policy matches |
| `fixed_stage` | `training_room` | Lock stage for all matches |

## Character selection

The `character_selection.mode` config field controls how characters are assigned:

| Mode | Description |
|---|---|
| `mirror` | Both players get the same random character (default) |
| `assigned` | Characters set explicitly via `assignments.p1` / `assignments.p2` |
| `random_from_pool` | Random pick from `pool` list |
| `llm_choice` | Each LLM chooses its own character via a dedicated prompt |

### LLM character choice

When `mode` is `llm_choice`, the daemon prompts each provider-backed policy to select a character before the match starts. Both players choose simultaneously (blind pick). Baseline policies fall back to seeded random selection.

The character selection prompt (`prompts/character_select_v1.md`) describes each character's archetype, strengths, weaknesses, and key moves. The LLM returns structured output with reasoning and a character name.

In tournament context, the prompt includes past match results against the current opponent (character picks, win/loss, final HP) so the LLM can adapt its character choice across a series.

Example config:

```json
{
  "character_selection": {
    "mode": "llm_choice"
  }
}
```

The mod receives concrete character assignments (as if mode were `assigned`) — no mod-side changes are needed.

## Analyzing results

### Artifact directory

Each completed match produces:

```
runs/<timestamp>_<match_id>/
  manifest.json       Config, versions, seed
  events.jsonl        Lifecycle events (MatchStarted, TurnRequested, etc.)
  decisions.jsonl     Per-turn request/decision pairs
  prompts.jsonl       Rendered prompts (when logging.prompts enabled)
  metrics.json        Latency stats, fallback rate, token usage
  result.json         Winner, end reason, turn count, status
  replay_index.json   Per-turn pointers into decisions/prompts
  stderr.log          Error output
  replay.mp4          Replay video (when replay_capture.enabled, default on)
  match.replay        Game replay file (when replay_capture.enabled, default on)
```

### Quick result check

```bash
cat runs/*/result.json | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f\"{r.get('match_id','?')[:8]}  winner={r.get('winner','?')}  reason={r.get('end_reason','?')}  turns={r.get('total_turns','?')}\")
"
```

### Tournament report

Generate a report from all completed runs:

```bash
scripts/run_round_robin.sh report --runs-dir runs/
```

Or filter to specific matches:

```bash
scripts/run_round_robin.sh report --match-ids match_abc match_def
```

The report includes:
- **Leaderboard** with Elo ratings, win/loss/draw counts, and win rates
- **Matchup table** showing head-to-head records
- **Latency summaries** per policy

Save to a file:

```bash
scripts/run_round_robin.sh report --output results/tournament_report.json
```

### Recomputing ratings

Ratings are derived from `result.json` artifacts and can be recomputed at any time with different parameters:

```bash
scripts/run_round_robin.sh report --k-factor 16
```

Mirror matches (same policy on both sides) are excluded from Elo calculations.

## Reproducibility

### Baseline runs

Baseline-only runs are fully deterministic. To reproduce a run exactly, reuse:

- `trace_seed` from the original `manifest.json`
- `match_id`
- Full config (policy_mapping, fallback_mode, decision_timeout_ms)

```bash
scripts/run_local_match.sh \
  --p1-policy baseline/random --p2-policy baseline/block_always \
  --trace-seed 42
```

The seed combines with `match_id`, `player_id`, `turn_id`, `state_hash`, and `legal_actions_hash` to derive per-turn RNG state.

### Provider-backed runs

Provider decisions are inherently non-deterministic due to model sampling. The `trace_seed` still controls fallback selection and tie-breaking, but primary decisions may vary across runs.

## Benchmarking

### Per-match metrics

Each run's `metrics.json` contains latency statistics, fallback counts, and token usage. Compare across runs to detect regressions.

Key metrics to watch:
- `average_latency_ms` — mean decision latency (baseline: <1ms, LLM: ~5-7s per concurrent pair)
- `fallback_count` — should be 0 for baseline, <5% for LLM
- `tokens_in_total` / `tokens_out_total` — total token usage for cost tracking

Note: decision requests for both players are processed concurrently. When both sides use LLM policies, two API calls run in parallel, so per-turn-pair latency is roughly the max of the two calls (~7s) rather than their sum (~14s).

## Prompt variants

Four prompt templates are available for provider-backed policies:

| Template | Use case |
|---|---|
| `minimal_v1` | Compact, fast, low token usage (default) |
| `strategic_v1` | Expanded strategic reasoning guidance |
| `few_shot_v1` | Includes worked examples of legal decisions |
| `reasoning_v1` | Encourages chain-of-thought reasoning |

Set per-policy in the config:

```json
{
  "anthropic/claude-strategic": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "prompt_version": "strategic_v1",
    "credential_env_var": "ANTHROPIC_API_KEY"
  }
}
```

## Correction retry

Provider policies can optionally retry once when the model returns invalid output:

```json
{
  "anthropic/claude-with-retry": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "prompt_version": "minimal_v1",
    "credential_env_var": "ANTHROPIC_API_KEY",
    "options": {
      "response_parser": {
        "enable_correction_retry": true,
        "max_correction_retries": 1
      }
    }
  }
}
```

When enabled, a bounded correction prompt is sent if the first response fails parsing with `malformed_output` or `illegal_output`. The second attempt is parsed normally. If it also fails, the turn falls back.

## Example experiment: comparing prompt strategies

1. Create a config with multiple policies using different prompts:

```json
{
  "policies": {
    "anthropic/minimal": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "prompt_version": "minimal_v1",
      "credential_env_var": "ANTHROPIC_API_KEY"
    },
    "anthropic/strategic": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "prompt_version": "strategic_v1",
      "credential_env_var": "ANTHROPIC_API_KEY"
    },
    "anthropic/reasoning": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "prompt_version": "reasoning_v1",
      "credential_env_var": "ANTHROPIC_API_KEY"
    },
    "baseline/greedy_damage": {
      "provider": "baseline",
      "prompt_version": "none"
    }
  },
  "tournament": {
    "format": "round_robin",
    "games_per_pair": 5,
    "side_swap": true
  }
}
```

2. Run the tournament:

```bash
ANTHROPIC_API_KEY=sk-ant-... scripts/run_round_robin.sh \
  --config daemon/config/prompt_experiment.json \
  anthropic/minimal anthropic/strategic anthropic/reasoning baseline/greedy_damage
```

3. Generate a report:

```bash
scripts/run_round_robin.sh report --output results/prompt_comparison.json
```

4. Compare Elo ratings, win rates, fallback rates, and latency across prompt strategies.

## DeepSeek verification

DeepSeek models can be used through the `openrouter` adapter because the DeepSeek API is OpenAI-compatible. Two sample configs are provided:

- `daemon/config/deepseek_vs_baseline.json` — standard 750 HP match
- `daemon/config/deepseek_vs_baseline_short.json` — 200 HP match for faster verification

Required environment variable:

```bash
DEEPSEEK_API_KEY=sk-...  # add to .env
```

Run a short verification match:

```bash
scripts/run_match_podman.sh --game-dir /path/to/YomiHustle \
  --daemon-config daemon/config/deepseek_vs_baseline_short.json
```

Or natively on Linux (requires Xvfb):

```bash
GAME_DIR=/path/to/YomiHustle \
  scripts/run_match_linux.sh --daemon-config daemon/config/deepseek_vs_baseline_short.json
```

Verified result (2026-06-29, deepseek-v4-flash vs baseline/random, 200 HP):

```text
Status:  completed
Winner:  p1 (DeepSeek)
Reason:  ko
Turns:   14
Character selection: P1=Cowboy, P2=Wizard
Fallback count: 0
Average latency: ~20 s/decision
Tokens: 36,360 in / 9,168 out
```

Notes:

- DeepSeek does not support `response_format: {type: "json_schema"}`, so set `"response_format": "json_object"` in policy options.
- Character selection (`llm_choice` mode) also respects `base_url` and `response_format` from policy options.
- DeepSeek API latency is higher than Anthropic/OpenAI; expect matches to take several minutes.
