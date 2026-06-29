"""Neural policy networks for RL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class YomiPolicy(nn.Module):
    """Masked categorical policy network for YOMI Hustle.

    Inputs:
        - observation vector (batch, obs_dim)
        - legal action mask (batch, action_dim) with 1.0 for legal actions

    Output:
        - action logits masked to legal actions
        - state value estimate (optional, for actor-critic methods)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden_dims: tuple[int, ...] = (512, 256),
        use_value_head: bool = False,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_value_head = use_value_head

        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.shared = nn.Sequential(*layers)

        self.action_head = nn.Linear(prev, action_dim)
        if use_value_head:
            self.value_head = nn.Linear(prev, 1)

    def forward(
        self, obs: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (masked action logits, value estimate or None)."""
        x = self.shared(obs)
        action_logits = self.action_head(x)

        # Mask illegal actions with a large negative value before softmax
        masked_logits = action_logits.masked_fill(legal_mask < 0.5, -1e9)

        value: torch.Tensor | None = None
        if self.use_value_head:
            value = self.value_head(x).squeeze(-1)

        return masked_logits, value

    def action_probs(
        self, obs: np.ndarray, legal_mask: np.ndarray
    ) -> tuple[np.ndarray, float | None]:
        """Numpy convenience method for inference."""
        self.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).float()
            mask_t = torch.from_numpy(legal_mask).unsqueeze(0).float()
            logits, value = self.forward(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            value_scalar = float(value.item()) if value is not None else None
            return probs, value_scalar

    def select_action(
        self,
        obs: np.ndarray,
        legal_mask: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> int:
        """Select an action index from a legal mask."""
        probs, _value = self.action_probs(obs, legal_mask)
        legal_indices = np.where(legal_mask > 0.5)[0]
        if len(legal_indices) == 0:
            raise ValueError("No legal actions available")

        if deterministic:
            idx = int(np.argmax(probs[legal_indices]))
        else:
            legal_probs = probs[legal_indices]
            legal_probs /= legal_probs.sum()
            idx = int(np.random.choice(len(legal_indices), p=legal_probs))

        return int(legal_indices[idx])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "use_value_head": self.use_value_head,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "YomiPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            use_value_head=bool(checkpoint.get("use_value_head", False)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
