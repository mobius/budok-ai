"""Behavior cloning trainer for YOMI Hustle RL policies."""

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
from torch.utils.data import DataLoader, Dataset

from yomi_daemon.rl import ActionEncoder, ObservationEncoder, TrajectoryExtractor
from yomi_daemon.rl.policy import YomiPolicy
from yomi_daemon.rl.types import TrajectoryStep

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """PyTorch dataset over trajectory steps."""

    def __init__(self, steps: Sequence[TrajectoryStep]) -> None:
        self.steps = list(steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        step = self.steps[idx]
        obs = torch.from_numpy(step.obs_vector).float()
        mask = torch.from_numpy(step.legal_action_mask).float()
        action = torch.tensor(step.action_index, dtype=torch.long)
        return obs, mask, action


def load_trajectories(
    runs_root: Path,
    *,
    max_matches: int | None = None,
    min_vocab_actions: int | None = None,
) -> tuple[list[TrajectoryStep], ObservationEncoder, ActionEncoder]:
    """Load all trajectory steps from match artifacts."""
    observation_encoder = ObservationEncoder()
    action_encoder = ActionEncoder()
    extractor = TrajectoryExtractor(observation_encoder, action_encoder)

    trajectories = extractor.extract_from_runs_root(runs_root, max_matches=max_matches)
    logger.info("Loaded %d trajectories", len(trajectories))

    steps: list[TrajectoryStep] = []
    for traj in trajectories:
        steps.extend(traj.steps)
    logger.info("Loaded %d steps", len(steps))

    if min_vocab_actions is not None and action_encoder.vocab_size < min_vocab_actions:
        logger.warning(
            "Action vocabulary only has %d actions; expected at least %d",
            action_encoder.vocab_size,
            min_vocab_actions,
        )

    return steps, observation_encoder, action_encoder


def train(
    steps: Sequence[TrajectoryStep],
    observation_encoder: ObservationEncoder,
    action_encoder: ActionEncoder,
    *,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    train_split: float = 0.9,
    seed: int = 42,
) -> dict[str, float]:
    """Train a behavior cloning policy and save checkpoints."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    all_steps = list(steps)
    if not all_steps:
        raise ValueError("No steps to train on")

    expected_size = len(all_steps[0].obs_vector)
    valid_steps = [s for s in all_steps if len(s.obs_vector) == expected_size]
    if len(valid_steps) != len(all_steps):
        logger.warning(
            "Filtered %d steps with non-uniform observation size (expected %d)",
            len(all_steps) - len(valid_steps),
            expected_size,
        )

    random.shuffle(valid_steps)
    split_idx = int(len(valid_steps) * train_split)
    train_steps = valid_steps[:split_idx]
    val_steps = valid_steps[split_idx:]

    logger.info("Train steps: %d, Val steps: %d", len(train_steps), len(val_steps))

    train_loader = DataLoader(
        TrajectoryDataset(train_steps),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TrajectoryDataset(val_steps),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    obs_dim = observation_encoder.vector_size
    action_dim = action_encoder.vocab_size
    model = YomiPolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_dims=(512, 256))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_acc = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for obs, mask, action in train_loader:
            optimizer.zero_grad()
            logits, _value = model(obs, mask)
            loss = F.cross_entropy(logits, action)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * obs.size(0)
            pred = logits.argmax(dim=-1)
            train_correct += (pred == action).sum().item()
            train_total += obs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        val_loss, val_acc = _evaluate(model, val_loader)
        logger.info(
            "Epoch %d/%d - train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(output_dir / "bc_best.pt")

    model.save(output_dir / "bc_final.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "encoder_config.json").write_text(
        json.dumps(
            {
                "observation_encoder": {
                    "version": observation_encoder.VERSION,
                    "history_len": observation_encoder.history_len,
                    "vector_size": observation_encoder.vector_size,
                },
                "action_encoder": {
                    "vocab_size": action_encoder.vocab_size,
                    "action_to_index": action_encoder.action_to_index,
                },
            },
            indent=2,
        )
    )

    return {
        "best_val_acc": best_val_acc,
        "final_val_acc": val_acc,
        "train_steps": len(train_steps),
        "val_steps": len(val_steps),
    }


def _evaluate(
    model: YomiPolicy,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for obs, mask, action in loader:
            logits, _value = model(obs, mask)
            loss = F.cross_entropy(logits, action)
            total_loss += loss.item() * obs.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == action).sum().item()
            total += obs.size(0)
    return total_loss / total if total else 0.0, correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a behavior cloning policy")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/bc"))
    parser.add_argument("--max-matches", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    steps, observation_encoder, action_encoder = load_trajectories(
        args.runs_root,
        max_matches=args.max_matches,
    )

    if not steps:
        logger.error("No trajectory steps found in %s", args.runs_root)
        raise SystemExit(1)

    metrics = train(
        steps,
        observation_encoder,
        action_encoder,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    logger.info("Training complete: %s", metrics)


if __name__ == "__main__":
    main()
