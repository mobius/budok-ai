"""RL environment that runs real YOMI Hustle matches as episodes."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from yomi_daemon.rl import ActionEncoder, ObservationEncoder, TrajectoryExtractor
from yomi_daemon.rl.types import TrajectoryStep


class MatchEnvironment:
    """Run a YOMI Hustle match and return a trajectory with dense rewards.

    This is a slow environment: each episode is one full match against a
    real game instance. It is intended for offline/batch RL rather than
    high-frequency sampling.
    """

    def __init__(
        self,
        *,
        game_dir: Path,
        observation_encoder: ObservationEncoder,
        action_encoder: ActionEncoder,
        opponent: str = "baseline/random",
        starting_hp: int = 200,
        stage_id: str = "training_room",
        character_mode: str = "mirror",
        use_podman: bool = False,
        no_replay: bool = True,
    ) -> None:
        self.game_dir = game_dir
        self.observation_encoder = observation_encoder
        self.action_encoder = action_encoder
        self.opponent = opponent
        self.starting_hp = starting_hp
        self.stage_id = stage_id
        self.character_mode = character_mode
        self.use_podman = use_podman
        self.no_replay = no_replay
        self._temp_dir: Path | None = None

    def run_episode(
        self,
        rl_model_path: Path,
        *,
        seed: int = 0,
    ) -> tuple[list[TrajectoryStep], dict[str, Any]]:
        """Run one match and return the RL-agent trajectory with rewards.

        Only the active player's steps are returned (the opponent steps are
        implicitly part of the environment transition).
        """
        config_path = self._write_config(rl_model_path, seed=seed)
        self._run_match(config_path)

        latest_run = self._find_latest_run()
        if latest_run is None:
            raise RuntimeError("No run directory found after match")

        result = json.loads((latest_run / "result.json").read_text(encoding="utf-8"))
        extractor = TrajectoryExtractor(self.observation_encoder, self.action_encoder)
        trajectory = extractor.extract_from_directory(latest_run)
        if trajectory is None:
            raise RuntimeError(f"Failed to extract trajectory from {latest_run}")

        rl_steps = self._compute_rewards(trajectory, result.get("winner"))
        return rl_steps, result

    def _write_config(self, rl_model_path: Path, *, seed: int) -> Path:
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="yomi_rl_"))

        encoder_dir = str(rl_model_path.parent)
        config: dict[str, Any] = {
            "version": "v1",
            "transport": {"host": "0.0.0.0", "port": 8765},
            "decision_timeout_ms": 10000,
            "fallback_mode": "safe_continue",
            "logging": {"events": True, "prompts": False, "raw_provider_payloads": False},
            "policy_mapping": {"p1": "rl/agent", "p2": self.opponent},
            "policies": {
                "rl/agent": {
                    "provider": "rl",
                    "model": "agent",
                    "prompt_version": "none",
                    "options": {
                        "model_path": str(rl_model_path),
                        "encoder_dir": encoder_dir,
                        "deterministic": False,
                    },
                },
            },
            "character_selection": {"mode": self.character_mode},
            "trace_seed": seed,
            "stage_id": self.stage_id,
            "match_options": {"starting_hp": self.starting_hp},
        }

        # Baseline opponents need their policy definitions added.
        if self.opponent == "baseline/random":
            config["policies"]["baseline/random"] = {"provider": "baseline", "prompt_version": "none"}
        elif self.opponent == "baseline/greedy_damage":
            config["policies"]["baseline/greedy_damage"] = {"provider": "baseline", "prompt_version": "none"}
        elif self.opponent == "baseline/scripted_safe":
            config["policies"]["baseline/scripted_safe"] = {"provider": "baseline", "prompt_version": "none"}

        config_path = self._temp_dir / "rl_train_config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        return config_path

    def _run_match(self, config_path: Path) -> None:
        # env.py is at daemon/src/yomi_daemon/rl/env.py; repo root is 4 parents up.
        repo_root = Path(__file__).resolve().parents[4]
        if self.use_podman:
            cmd = [
                str(repo_root / "scripts" / "run_match_podman.sh"),
                "--game-dir",
                str(self.game_dir),
                "--daemon-config",
                str(config_path),
            ]
            if self.no_replay:
                cmd.append("--no-replay")
        else:
            cmd = [
                str(repo_root / "scripts" / "run_match_linux.sh"),
                "--daemon-config",
                str(config_path),
            ]
            if self.no_replay:
                cmd.append("--no-replay")
            env = {"GAME_DIR": str(self.game_dir)}

        subprocess.run(
            cmd,
            cwd=repo_root,
            env=env if not self.use_podman else None,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _find_latest_run(self) -> Path | None:
        repo_root = Path(__file__).resolve().parents[4]
        runs_root = repo_root / "runs"
        if not runs_root.exists():
            return None
        dirs = [d for d in runs_root.iterdir() if d.is_dir()]
        if not dirs:
            return None
        return max(dirs, key=lambda d: d.stat().st_mtime)

    def _compute_rewards(
        self,
        trajectory: Any,
        winner: str | None,
    ) -> list[TrajectoryStep]:
        """Compute dense rewards from HP changes and attach next_obs/done."""
        # Group steps by player and sort by turn
        p1_steps: list[TrajectoryStep] = []
        p2_steps: list[TrajectoryStep] = []
        for step in trajectory.steps:
            if step.player_id == "p1":
                p1_steps.append(step)
            else:
                p2_steps.append(step)

        p1_steps.sort(key=lambda s: s.turn_id)
        p2_steps.sort(key=lambda s: s.turn_id)

        def hp_from_obs(obs: np.ndarray) -> float:
            # Fighter 0 (p1) HP is at index 1 (after active_player)
            return float(obs[1])

        def build_rl_steps(agent_steps: list[TrajectoryStep], opponent_steps: list[TrajectoryStep]) -> list[TrajectoryStep]:
            # Find previous HP for agent and opponent from consecutive observations
            steps_out: list[TrajectoryStep] = []
            for i, step in enumerate(agent_steps):
                agent_hp = hp_from_obs(step.obs_vector)
                # Opponent HP is at index 1 + fighter_vec_size
                opponent_hp_idx = 1 + self.observation_encoder.fighter_vec_size
                opponent_hp = float(step.obs_vector[opponent_hp_idx])

                next_obs = agent_steps[i + 1].obs_vector if i + 1 < len(agent_steps) else None
                next_agent_hp = hp_from_obs(next_obs) if next_obs is not None else agent_hp
                next_opponent_hp = float(next_obs[opponent_hp_idx]) if next_obs is not None else opponent_hp

                agent_hp_delta = next_agent_hp - agent_hp
                opponent_hp_delta = next_opponent_hp - opponent_hp

                reward = -agent_hp_delta * 0.01 + opponent_hp_delta * 0.01

                is_last = i == len(agent_steps) - 1
                done = is_last and winner is not None
                if is_last and winner is not None:
                    if winner == step.player_id:
                        reward += 1.0
                    else:
                        reward -= 1.0

                next_mask = agent_steps[i + 1].legal_action_mask if i + 1 < len(agent_steps) else None

                steps_out.append(
                    TrajectoryStep(
                        match_id=step.match_id,
                        turn_id=step.turn_id,
                        player_id=step.player_id,
                        obs_vector=step.obs_vector,
                        legal_action_mask=step.legal_action_mask,
                        action_index=step.action_index,
                        reward=reward,
                        next_obs_vector=next_obs,
                        next_legal_action_mask=next_mask,
                        done=done,
                        info=step.info,
                    )
                )
            return steps_out

        # Return steps for p1 (RL agent)
        return build_rl_steps(p1_steps, p2_steps)

    def cleanup(self) -> None:
        if self._temp_dir is not None and self._temp_dir.exists():
            shutil.rmtree(self._temp_dir)
            self._temp_dir = None
