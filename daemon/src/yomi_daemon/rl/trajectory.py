"""Extract RL trajectories from match artifact directories."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from yomi_daemon.rl.action_encoder import ActionEncoder
from yomi_daemon.rl.observation_encoder import ObservationEncoder
from yomi_daemon.rl.types import MatchTrajectory, TrajectoryStep

logger = logging.getLogger(__name__)


class TrajectoryExtractor:
    """Convert a directory of match artifacts into RL trajectories."""

    def __init__(
        self,
        observation_encoder: ObservationEncoder | None = None,
        action_encoder: ActionEncoder | None = None,
    ) -> None:
        self.observation_encoder = observation_encoder or ObservationEncoder()
        self.action_encoder = action_encoder or ActionEncoder()

    def extract_from_directory(self, run_dir: Path) -> MatchTrajectory | None:
        """Extract a MatchTrajectory from a single run directory."""
        result_path = run_dir / "result.json"
        decisions_path = run_dir / "decisions.jsonl"
        events_path = run_dir / "events.jsonl"

        if not result_path.exists() or not decisions_path.exists():
            return None

        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed":
            logger.debug("Skipping incomplete match: %s", run_dir.name)
            return None

        decisions = _read_jsonl(decisions_path)
        events = _read_jsonl(events_path) if events_path.exists() else []

        return self._build_trajectory(result, decisions, events)

    def extract_from_runs_root(
        self,
        runs_root: Path,
        *,
        max_matches: int | None = None,
    ) -> list[MatchTrajectory]:
        """Extract trajectories from all completed matches under ``runs_root``."""
        trajectories: list[MatchTrajectory] = []
        for run_dir in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            traj = self.extract_from_directory(run_dir)
            if traj is not None:
                trajectories.append(traj)
                if max_matches is not None and len(trajectories) >= max_matches:
                    break
        return trajectories

    def _build_trajectory(
        self,
        result: Mapping[str, Any],
        decisions: Sequence[Mapping[str, Any]],
        events: Sequence[Mapping[str, Any]],
    ) -> MatchTrajectory:
        match_id = str(result.get("match_id", "unknown"))
        p1_policy = ""
        p2_policy = ""
        for event in events:
            payload = event.get("payload", {})
            details = payload.get("details", {})
            if payload.get("event") == "MatchStarted":
                p1_policy = str(details.get("p1_policy", ""))
                p2_policy = str(details.get("p2_policy", ""))
                break

        # Group decisions by turn_id and compute per-step rewards
        steps: list[TrajectoryStep] = []
        turn_decisions: dict[int, dict[str, Mapping[str, Any]]] = {}
        for decision in decisions:
            turn_id = int(decision.get("turn_id", 0))
            player_id = str(decision.get("player_id", "p1"))
            turn_decisions.setdefault(turn_id, {})[player_id] = decision

        sorted_turns = sorted(turn_decisions.keys())
        for i, turn_id in enumerate(sorted_turns):
            turn = turn_decisions[turn_id]
            for player_id in ("p1", "p2"):
                if player_id not in turn:
                    continue
                decision = turn[player_id]
                step = self._build_step(
                    match_id=match_id,
                    turn_id=turn_id,
                    player_id=player_id,
                    decision=decision,
                    is_last_turn=(i == len(sorted_turns) - 1),
                    result=result,
                )
                if step is not None:
                    steps.append(step)

        return MatchTrajectory(
            match_id=match_id,
            p1_policy=p1_policy,
            p2_policy=p2_policy,
            winner=result.get("winner"),
            end_reason=result.get("end_reason"),
            total_turns=int(result.get("total_turns", 0)),
            steps=steps,
        )

    def _build_step(
        self,
        *,
        match_id: str,
        turn_id: int,
        player_id: str,
        decision: Mapping[str, Any],
        is_last_turn: bool,
        result: Mapping[str, Any],
    ) -> TrajectoryStep | None:
        request = decision.get("request_payload", {})
        observation = request.get("observation")
        legal_actions = request.get("legal_actions", [])
        decision_payload = decision.get("decision_payload", {})

        if observation is None or not legal_actions:
            return None

        # Ensure action vocabulary includes everything in this match
        legal_indices = self.action_encoder.encode_legal_actions(legal_actions)
        action_name = str(decision_payload.get("action", ""))
        action_data = decision_payload.get("data")
        action_index = self.action_encoder.encode_action(action_name, action_data)

        obs_vector = self.observation_encoder.encode(observation)
        legal_mask, _ = self.action_encoder.build_mask(legal_indices)

        # Compute reward: outcome at end, otherwise sparse 0
        reward = 0.0
        done = False
        if is_last_turn and result.get("winner") == player_id:
            reward = 1.0
            done = True
        elif is_last_turn and result.get("winner") is not None:
            reward = -1.0
            done = True

        return TrajectoryStep(
            match_id=match_id,
            turn_id=turn_id,
            player_id=player_id,
            obs_vector=obs_vector,
            legal_action_mask=legal_mask,
            action_index=action_index,
            reward=reward,
            done=done,
            info={
                "action_name": action_name,
                "policy_id": decision_payload.get("policy_id", ""),
            },
        )


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records
