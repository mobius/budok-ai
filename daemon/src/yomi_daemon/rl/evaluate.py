"""Evaluate a trained RL policy against baseline opponents."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from yomi_daemon.rl import ActionEncoder, ObservationEncoder
from yomi_daemon.rl.env import MatchEnvironment
from yomi_daemon.rl.policy import YomiPolicy

logger = logging.getLogger(__name__)


def evaluate(
    *,
    game_dir: Path,
    model_path: Path,
    encoder_dir: Path,
    opponents: list[str],
    matches_per_opponent: int,
    starting_hp: int,
    use_podman: bool,
    seed: int = 1000,
) -> dict[str, dict[str, float]]:
    """Run evaluation matches and return per-opponent win rates."""
    observation_encoder = _load_observation_encoder(encoder_dir)
    action_encoder = _load_action_encoder(encoder_dir)

    policy = YomiPolicy.load(model_path)
    policy.eval()

    results: dict[str, dict[str, float]] = {}

    for opponent in opponents:
        wins = 0
        losses = 0
        draws = 0
        total_turns = 0

        env = MatchEnvironment(
            game_dir=game_dir,
            observation_encoder=observation_encoder,
            action_encoder=action_encoder,
            opponent=opponent,
            starting_hp=starting_hp,
            use_podman=use_podman,
            no_replay=True,
        )

        try:
            for i in range(matches_per_opponent):
                match_seed = seed + i
                logger.info("Evaluating %s vs %s (match %d/%d)", model_path.name, opponent, i + 1, matches_per_opponent)
                steps, result = env.run_episode(model_path, seed=match_seed)
                winner = result.get("winner")
                total_turns += int(result.get("total_turns") or 0)
                if winner == "p1":
                    wins += 1
                elif winner == "p2":
                    losses += 1
                else:
                    draws += 1
        finally:
            env.cleanup()

        total = wins + losses + draws
        results[opponent] = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "win_rate": wins / total if total else 0.0,
            "avg_turns": total_turns / matches_per_opponent if matches_per_opponent else 0.0,
        }

    return results


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
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy")
    parser.add_argument("--game-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", default=["baseline/random", "baseline/greedy_damage", "baseline/scripted_safe"])
    parser.add_argument("--matches", type=int, default=3)
    parser.add_argument("--starting-hp", type=int, default=200)
    parser.add_argument("--use-podman", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    results = evaluate(
        game_dir=args.game_dir,
        model_path=args.model,
        encoder_dir=args.encoder_dir,
        opponents=args.opponents,
        matches_per_opponent=args.matches,
        starting_hp=args.starting_hp,
        use_podman=args.use_podman,
        seed=args.seed,
    )

    print("\n=== Evaluation Results ===")
    for opponent, metrics in results.items():
        print(
            f"{opponent}: wins={metrics['wins']}/{args.matches} "
            f"({metrics['win_rate']*100:.1f}%), avg_turns={metrics['avg_turns']:.1f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
