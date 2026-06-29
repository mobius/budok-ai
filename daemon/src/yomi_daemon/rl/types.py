"""Shared types for the RL subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """A single step in an RL trajectory.

    Fields mirror an MDP transition: (s, a, r, s', done, info).
    """

    match_id: str
    turn_id: int
    player_id: str
    obs_vector: np.ndarray
    legal_action_mask: np.ndarray
    action_index: int
    reward: float
    next_obs_vector: np.ndarray | None = None
    next_legal_action_mask: np.ndarray | None = None
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MatchTrajectory:
    """All steps for both players in a single match."""

    match_id: str
    p1_policy: str
    p2_policy: str
    winner: str | None
    end_reason: str | None
    total_turns: int
    steps: list[TrajectoryStep] = field(default_factory=list)
