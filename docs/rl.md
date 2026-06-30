# Reinforcement Learning Guide

This document describes the RL subsystem added to budok-ai for training neural-network policies that play YOMI Hustle.

## Overview

The RL pipeline has four phases:

1. **Data / feature engineering** — extract trajectories from match artifacts
2. **Behavior cloning (BC)** — warm-start a policy from strong player data
3. **PPO self-play** — fine-tune the policy using match outcomes
4. **Deployment / evaluation** — serve the trained policy via the `rl` adapter

## Module layout

```
daemon/src/yomi_daemon/rl/
├── __init__.py              # Public exports
├── action_encoder.py        # Discrete action vocabulary and legal-action masking
├── env.py                   # MatchEnvironment: runs real matches as RL episodes
├── evaluate.py              # Win-rate evaluation against baseline opponents
├── observation_encoder.py   # Observation JSON -> fixed-size vector
├── policy.py                # YomiPolicy PyTorch MLP
├── train_bc.py              # Behavior cloning trainer
├── train_ppo.py             # PPO trainer with GAE
└── types.py                 # Trajectory data structures
```

## Observation encoding

`ObservationEncoder` converts the daemon observation JSON into a fixed-size float32 vector. The vector includes:

- Active player id
- Per-fighter continuous features (HP, meter, position, velocity, combo count, ...)
- Per-fighter boolean features (grounded, initiative, can_feint, ...)
- Per-fighter categorical features (character, facing)
- Stage features (width, ceiling height, has ceiling)
- Character-specific data aggregates
- Recent turn history (actions, HP deltas)

The encoder is versioned so that a trained policy is not silently broken by observation schema changes.

## Action encoding

`ActionEncoder` builds a discrete vocabulary of `(action_name, canonical_payload)` pairs. Payload values with countable parameters are bucketed to keep the action space small. During inference the policy receives a binary mask over its full vocabulary and masks illegal actions before the softmax.

## Phase 1: Behavior cloning

Generate or reuse match data, then train a supervised policy to predict the chosen action.

### Generate baseline dataset

```bash
scripts/generate_bc_dataset.sh \
  --game-dir /path/to/YomiHustle \
  --count 10
```

This runs matches from these configs:

- `daemon/config/bc_scripted_vs_random.json`
- `daemon/config/bc_greedy_vs_random.json`
- `daemon/config/bc_greedy_vs_scripted.json`

Artifacts are written to `runs/`.

### Train BC model

```bash
cd budok-ai
uv run --project daemon python3 -m yomi_daemon.rl.train_bc \
  --runs-root runs \
  --output-dir models/bc_baseline \
  --epochs 30 \
  --batch-size 64
```

Outputs:

- `models/bc_baseline/bc_best.pt` — best validation checkpoint
- `models/bc_baseline/bc_final.pt` — final checkpoint
- `models/bc_baseline/encoder_config.json` — observation/action encoder config

### Evaluate BC model

```bash
uv run --project daemon python3 -m yomi_daemon.rl.evaluate \
  --game-dir /path/to/YomiHustle \
  --model models/bc_baseline/bc_best.pt \
  --encoder-dir models/bc_baseline \
  --opponents baseline/random baseline/greedy_damage \
  --matches 5 \
  --starting-hp 200 \
  --use-podman
```

## Phase 2: PPO self-play

PPO uses the BC model as a warm start and plays real matches against baseline opponents. Each episode is one full match; rewards are computed from HP deltas and match outcome.

```bash
uv run --project daemon python3 -m yomi_daemon.rl.train_ppo \
  --game-dir /path/to/YomiHustle \
  --bc-model models/bc_baseline/bc_best.pt \
  --total-episodes 50 \
  --episodes-per-batch 4 \
  --starting-hp 200 \
  --opponent baseline/random \
  --use-podman \
  --output-dir models/ppo
```

Outputs:

- `models/ppo/ppo_batch_*.pt` — per-batch checkpoints
- `models/ppo/ppo_best.pt` — checkpoint with highest batch win rate
- `models/ppo/ppo_final.pt` — final checkpoint
- `models/ppo/history.json` — training metrics

### Important PPO notes

- Episode collection is **slow** because each episode runs the real game.
- Small batches and high learning rates can degrade a BC warm-start.
- Recommended starting point: `--total-episodes 50 --episodes-per-batch 8 --starting-hp 200`.
- For faster iteration, keep `--starting-hp 200`; increase to 750 only after the policy is stable.

## Phase 3: Deploy the RL policy

Add an `rl` policy to a daemon config:

```json
{
  "policies": {
    "rl/ppo_best": {
      "provider": "rl",
      "model": "ppo_best",
      "prompt_version": "none",
      "options": {
        "model_path": "models/ppo/ppo_best.pt",
        "encoder_dir": "models/bc_baseline",
        "deterministic": false
      }
    }
  }
}
```

Then map a player to it:

```json
{
  "policy_mapping": {
    "p1": "rl/ppo_best",
    "p2": "baseline/random"
  }
}
```

Run the match:

```bash
scripts/run_match_podman.sh \
  --game-dir /path/to/YomiHustle \
  --daemon-config daemon/config/ppo_vs_random_short.json
```

## Current results

These results are from a small proof-of-concept run and are intended to validate the pipeline, not represent a fully trained agent.

| Model | Opponent | Matches | Win rate | Notes |
|---|---|---|---|---|
| BC v2 | baseline/random | 3 | 66.7% | Warm-start from scripted/greedy/baseline data |
| PPO best | baseline/random | 2 | 0.0% | 8 episodes, small batch; policy degraded |
| PPO best | baseline/greedy_damage | 2 | 0.0% | Needs larger batches and tuning |

## Known limitations and next steps

1. **Action space is large** (~200+ discrete actions). Consider predicting action name only and using a fixed default payload.
2. **PPO is sample-inefficient** with real-game episodes. Each episode takes 1–5 minutes.
3. **Reward shaping** can be improved: add penalties for timeouts and for taking damage.
4. **Self-play** is not yet implemented; currently PPO trains against static baselines.
5. **Value network** is added after BC warm-start and may need more updates.

## Useful commands

```bash
# Train BC
uv run --project daemon python3 -m yomi_daemon.rl.train_bc \
  --runs-root runs --output-dir models/bc_baseline

# Train PPO
uv run --project daemon python3 -m yomi_daemon.rl.train_ppo \
  --game-dir /path/to/YomiHustle \
  --bc-model models/bc_baseline/bc_best.pt \
  --use-podman

# Evaluate
uv run --project daemon python3 -m yomi_daemon.rl.evaluate \
  --game-dir /path/to/YomiHustle \
  --model models/ppo/ppo_best.pt \
  --encoder-dir models/bc_baseline \
  --use-podman
```
