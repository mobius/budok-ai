"""PPO self-play trainer for YOMI Hustle RL policies."""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from yomi_daemon.rl import ActionEncoder, ObservationEncoder
from yomi_daemon.rl.env import MatchEnvironment
from yomi_daemon.rl.policy import YomiPolicy
from yomi_daemon.rl.types import TrajectoryStep

logger = logging.getLogger(__name__)


class PPOBuffer:
    """Simple rollout buffer for PPO."""

    def __init__(self) -> None:
        self.obs: list[np.ndarray] = []
        self.legal_masks: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []

    def add(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        self.obs.append(obs)
        self.legal_masks.append(legal_mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self) -> None:
        self.obs.clear()
        self.legal_masks.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.obs)


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    dones: Sequence[bool],
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
    next_value: float = 0.0,
) -> tuple[list[float], list[float]]:
    """Compute GAE advantages and discounted returns."""
    advantages: list[float] = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * (1.0 - float(dones[t])) - values[t]
        gae = delta + gamma * lam * (1.0 - float(dones[t])) * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns


def collect_episode(
    env: MatchEnvironment,
    policy: YomiPolicy,
    *,
    seed: int = 0,
) -> tuple[list[TrajectoryStep], dict[str, object]]:
    """Run one episode and return trajectory steps + match result."""
    # Save the model inside the repo so the Podman container can read it through
    # the repository volume mount.
    repo_root = Path(__file__).resolve().parents[4]
    rl_dir = repo_root / ".rl_train"
    rl_dir.mkdir(exist_ok=True)
    model_path = rl_dir / "policy.pt"
    policy.save(model_path)
    steps, result = env.run_episode(model_path, seed=seed)
    return steps, result


def train_ppo(
    *,
    env: MatchEnvironment,
    policy: YomiPolicy,
    output_dir: Path,
    total_episodes: int = 50,
    episodes_per_batch: int = 4,
    epochs_per_batch: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    seed: int = 42,
) -> dict[str, Any]:
    """Run PPO training with match episodes."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure value head exists before creating optimizer
    if getattr(policy, "value_head", None) is None:
        hidden_dim = policy.shared[-2].out_features
        policy.value_head = torch.nn.Linear(hidden_dim, 1)
        policy.use_value_head = True

    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    buffer = PPOBuffer()
    history: list[dict[str, float]] = []
    best_win_rate = -1.0

    for episode_start in range(0, total_episodes, episodes_per_batch):
        batch_idx = episode_start // episodes_per_batch
        buffer.clear()
        wins = 0
        losses = 0
        total_turns = 0

        for i in range(episodes_per_batch):
            episode_seed = seed + episode_start + i
            episode_num = episode_start + i + 1
            logger.info("Collecting episode %d/%d", episode_num, total_episodes)

            steps, result = collect_episode(env, policy, seed=episode_seed)
            total_turns += int(result.get("total_turns", 0))
            winner = result.get("winner")
            if winner == "p1":
                wins += 1
            elif winner is not None:
                losses += 1

            # Compute values and log_probs for each step
            values: list[float] = []
            log_probs: list[float] = []
            policy.eval()
            with torch.no_grad():
                for step in steps:
                    obs_t = torch.from_numpy(step.obs_vector).unsqueeze(0).float()
                    mask_t = torch.from_numpy(step.legal_action_mask).unsqueeze(0).float()
                    logits, value = policy(obs_t, mask_t)
                    dist = torch.distributions.Categorical(logits=F.log_softmax(logits, dim=-1))
                    log_prob = dist.log_prob(torch.tensor([step.action_index]))
                    values.append(float(value.item()))
                    log_probs.append(float(log_prob.item()))

            # Compute GAE
            next_value = 0.0
            advantages, returns = compute_gae(
                [s.reward for s in steps],
                values,
                [s.done for s in steps],
                gamma=gamma,
                lam=lam,
                next_value=next_value,
            )

            for step, advantage, ret, value, log_prob in zip(
                steps, advantages, returns, values, log_probs
            ):
                buffer.add(
                    obs=step.obs_vector,
                    legal_mask=step.legal_action_mask,
                    action=step.action_index,
                    reward=step.reward,
                    value=value,
                    log_prob=log_prob,
                    done=step.done,
                )

        if len(buffer) == 0:
            logger.warning("No steps collected in batch %d", batch_idx)
            continue

        # Update policy and value network
        policy.train()
        obs_t = torch.from_numpy(np.stack(buffer.obs)).float()
        mask_t = torch.from_numpy(np.stack(buffer.legal_masks)).float()
        actions_t = torch.tensor(buffer.actions, dtype=torch.long)
        old_log_probs_t = torch.tensor(buffer.log_probs, dtype=torch.float32)
        advantages_list, returns_list = compute_gae(
            buffer.rewards, buffer.values, buffer.dones, gamma=gamma, lam=lam
        )
        returns_t = torch.tensor(returns_list, dtype=torch.float32)
        advantages_t = torch.tensor(advantages_list, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_loss = 0.0
        for _ in range(epochs_per_batch):
            logits, values_pred = policy(obs_t, mask_t)
            dist = torch.distributions.Categorical(logits=F.log_softmax(logits, dim=-1))
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values_pred.squeeze(-1), returns_t)

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / epochs_per_batch
        win_rate = wins / episodes_per_batch if episodes_per_batch else 0.0
        avg_turns = total_turns / episodes_per_batch if episodes_per_batch else 0.0

        logger.info(
            "Batch %d: episodes=%d, steps=%d, wins=%d, losses=%d, win_rate=%.2f, avg_turns=%.1f, loss=%.4f",
            batch_idx,
            episodes_per_batch,
            len(buffer),
            wins,
            losses,
            win_rate,
            avg_turns,
            avg_loss,
        )
        history.append(
            {
                "batch": batch_idx,
                "win_rate": win_rate,
                "avg_turns": avg_turns,
                "loss": avg_loss,
                "episodes": episode_start + episodes_per_batch,
            }
        )

        policy.save(output_dir / f"ppo_batch_{batch_idx:03d}.pt")
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            policy.save(output_dir / "ppo_best.pt")

    policy.save(output_dir / "ppo_final.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))

    return {"best_win_rate": best_win_rate, "total_episodes": total_episodes}


def _load_observation_encoder(encoder_dir: Path) -> ObservationEncoder:
    config_path = encoder_dir / "encoder_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        obs_config = config.get("observation_encoder", {})
        return ObservationEncoder(
            history_len=int(obs_config.get("history_len", 3)),
            include_character_data=bool(obs_config.get("include_character_data", True)),
        )
    return ObservationEncoder()


def _load_action_encoder(encoder_dir: Path) -> ActionEncoder:
    config_path = encoder_dir / "encoder_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        action_to_index = config.get("action_encoder", {}).get("action_to_index")
        if isinstance(action_to_index, dict):
            encoder = ActionEncoder()
            for action_key, idx in sorted(action_to_index.items(), key=lambda kv: kv[1]):
                if ":" in action_key:
                    action_name, payload_json = action_key.split(":", 1)
                    payload = json.loads(payload_json)
                else:
                    action_name = action_key
                    payload = None
                encoder.encode_action(action_name, payload)
            return encoder
    return ActionEncoder()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOMI Hustle policy with PPO")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--bc-model", type=Path, default=None, help="Path to warm-start BC model")
    parser.add_argument("--output-dir", type=Path, default=Path("models/ppo"))
    parser.add_argument("--total-episodes", type=int, default=20)
    parser.add_argument("--episodes-per-batch", type=int, default=4)
    parser.add_argument("--starting-hp", type=int, default=200)
    parser.add_argument("--opponent", type=str, default="baseline/random")
    parser.add_argument("--use-podman", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.bc_model is not None and args.bc_model.exists():
        encoder_dir = args.bc_model.parent
        observation_encoder = _load_observation_encoder(encoder_dir)
        action_encoder = _load_action_encoder(encoder_dir)
    else:
        observation_encoder = ObservationEncoder()
        action_encoder = ActionEncoder()

    if args.bc_model is not None and args.bc_model.exists():
        logger.info("Loading warm-start model from %s", args.bc_model)
        policy = YomiPolicy.load(args.bc_model)
        policy.use_value_head = True
        # Add value head if not present
        if getattr(policy, "value_head", None) is None:
            hidden_dim = policy.shared[-2].out_features
            policy.value_head = torch.nn.Linear(hidden_dim, 1)
    else:
        logger.info("Initializing new policy")
        policy = YomiPolicy(
            obs_dim=observation_encoder.vector_size,
            action_dim=action_encoder.vocab_size,
            hidden_dims=(512, 256),
            use_value_head=True,
        )

    env = MatchEnvironment(
        game_dir=args.game_dir,
        observation_encoder=observation_encoder,
        action_encoder=action_encoder,
        opponent=args.opponent,
        starting_hp=args.starting_hp,
        use_podman=args.use_podman,
        no_replay=True,
    )

    try:
        metrics = train_ppo(
            env=env,
            policy=policy,
            output_dir=args.output_dir,
            total_episodes=args.total_episodes,
            episodes_per_batch=args.episodes_per_batch,
            seed=args.seed,
        )
        logger.info("PPO training complete: %s", metrics)
    finally:
        env.cleanup()


if __name__ == "__main__":
    main()
