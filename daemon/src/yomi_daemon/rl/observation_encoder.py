"""Observation encoder: converts daemon observation JSON to fixed-length vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from yomi_daemon.prompt import VALID_CHARACTERS


# Categorical vocabularies ----------------------------------------------------

CHARACTER_TO_INDEX = {name: i for i, name in enumerate(VALID_CHARACTERS)}
FACING_TO_INDEX = {"left": 0, "right": 1}
PLAYER_TO_INDEX = {"p1": 0, "p2": 1}

# Character-specific numeric fields we know about. Unknown fields are flattened
# dynamically at encoding time and averaged into a single numeric value.
KNOWN_CHARACTER_DATA_KEYS: dict[str, set[str]] = {
    "Cowboy": {"bullets_left", "consecutive_shots", "has_gun"},
    "Wizard": {"geyser_charge", "gusts_in_combo", "hover_left"},
    "Robot": {"steam_pressure", "gears", "overdrive"},
    "Ninja": {"kunai", "mark_stacks", "teleport_charges"},
    "Mutant": {"toxin", "tentacles", "rage"},
}


class ObservationEncoder:
    """Encode a daemon ``observation`` JSON object into a fixed-length vector.

    The encoder is deterministic and versioned so that trained policies are not
    silently broken by observation schema changes.
    """

    VERSION = 1

    def __init__(
        self,
        *,
        history_len: int = 3,
        include_character_data: bool = True,
    ) -> None:
        self.history_len = history_len
        self.include_character_data = include_character_data

        # Per-fighter continuous features (always present)
        self.fighter_continuous = [
            "hp",
            "max_hp",
            "meter",
            "burst",
            "pos_x",
            "pos_y",
            "vel_x",
            "vel_y",
            "combo_count",
            "blockstun",
            "hitlag",
            "combo_proration",
        ]
        # Per-fighter boolean features
        self.fighter_bools = [
            "state_interruptable",
            "can_feint",
            "grounded",
            "initiative",
        ]
        # Categorical per-fighter features
        self.fighter_categorical = ["character", "facing"]

        # Stage features
        self.stage_features = ["stage_width", "stage_ceiling_height", "has_ceiling"]

        # Fixed vector layout:
        # [active_player] + [fighter0_continuous + bools + cat] + [fighter1_continuous + bools + cat]
        # + [stage] + [character_data_0] + [character_data_1]
        # + [history_len * (action_cat + hp_delta * 2)]
        self.fighter_vec_size = (
            len(self.fighter_continuous)
            + len(self.fighter_bools)
            + len(self.fighter_categorical)
        )
        self.character_data_size = 4 if include_character_data else 0
        self.history_feature_size = 5  # p1_action, p2_action, p1_hp_delta, p2_hp_delta, was_fallback

        self.vector_size = (
            1  # active player
            + 2 * self.fighter_vec_size
            + len(self.stage_features)
            + 2 * self.character_data_size
            + self.history_len * self.history_feature_size
        )

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return a 1-D float32 vector for the observation."""
        vec = np.zeros(self.vector_size, dtype=np.float32)
        idx = 0

        # Active player
        active = str(observation.get("active_player", "p1"))
        vec[idx] = PLAYER_TO_INDEX.get(active, 0)
        idx += 1

        fighters = list(observation.get("fighters", []))
        if len(fighters) != 2:
            raise ValueError(f"Expected 2 fighters, got {len(fighters)}")

        # Sort fighters by id so p1 is always first
        fighters_by_id = {str(f["id"]): f for f in fighters}
        ordered = [fighters_by_id.get("p1"), fighters_by_id.get("p2")]
        if None in ordered:
            # Fallback to original order if ids are unexpected
            ordered = fighters[:2]

        for fighter in ordered:
            idx = self._encode_fighter(vec, idx, fighter)

        # Stage
        stage = observation.get("stage", {})
        vec[idx] = float(stage.get("width", 1100))
        idx += 1
        vec[idx] = float(stage.get("ceiling_height", 400))
        idx += 1
        vec[idx] = 1.0 if stage.get("has_ceiling", True) else 0.0
        idx += 1

        # Character data
        if self.include_character_data:
            for fighter in ordered:
                idx = self._encode_character_data(vec, idx, fighter)

        # History
        history = list(observation.get("history", []))
        idx = self._encode_history(vec, idx, history)

        return vec

    def _encode_fighter(
        self, vec: np.ndarray, idx: int, fighter: Mapping[str, Any]
    ) -> int:
        # Continuous
        pos = fighter.get("position", {})
        vel = fighter.get("velocity", {})
        values = [
            float(fighter.get("hp", 0)),
            float(fighter.get("max_hp", 0)),
            float(fighter.get("meter", 0)),
            float(fighter.get("burst", 0)),
            float(pos.get("x", 0.0)),
            float(pos.get("y", 0.0)),
            float(vel.get("x", 0.0)),
            float(vel.get("y", 0.0)),
            float(fighter.get("combo_count", 0)),
            float(fighter.get("blockstun", 0)),
            float(fighter.get("hitlag", 0)),
            float(fighter.get("combo_proration", 0.0)),
        ]
        for v in values:
            vec[idx] = v
            idx += 1

        # Bools
        for key in self.fighter_bools:
            vec[idx] = 1.0 if fighter.get(key, False) else 0.0
            idx += 1

        # Categoricals
        character = str(fighter.get("character", "Cowboy"))
        vec[idx] = CHARACTER_TO_INDEX.get(character, 0)
        idx += 1

        facing = str(fighter.get("facing", "right"))
        vec[idx] = FACING_TO_INDEX.get(facing, 1)
        idx += 1

        return idx

    def _encode_character_data(
        self, vec: np.ndarray, idx: int, fighter: Mapping[str, Any]
    ) -> int:
        """Flatten character-specific data into a small fixed-size vector."""
        data = fighter.get("character_data", {})
        if not isinstance(data, Mapping):
            return idx + self.character_data_size

        character = str(fighter.get("character", ""))
        known_keys = KNOWN_CHARACTER_DATA_KEYS.get(character, set())

        numeric_values: list[float] = []
        for key, value in data.items():
            if isinstance(value, bool):
                numeric_values.append(1.0 if value else 0.0)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_values.append(float(value))

        # Aggregate into fixed slots: mean, max, min, count of known fields.
        if numeric_values:
            arr = np.array(numeric_values, dtype=np.float32)
            vec[idx] = float(np.mean(arr))
            vec[idx + 1] = float(np.max(arr))
            vec[idx + 2] = float(np.min(arr))
            vec[idx + 3] = float(len(arr))
        idx += self.character_data_size
        return idx

    def _encode_history(
        self,
        vec: np.ndarray,
        idx: int,
        history: Sequence[Mapping[str, Any]],
    ) -> int:
        """Encode the most recent ``history_len`` history entries."""
        # Pad with zeros if history is short
        recent = list(history[-self.history_len :])
        while len(recent) < self.history_len:
            recent.insert(0, {})

        for entry in recent:
            p1_action = str(entry.get("p1_action", ""))
            p2_action = str(entry.get("p2_action", ""))
            vec[idx] = hash(p1_action) % 1000 / 1000.0
            idx += 1
            vec[idx] = hash(p2_action) % 1000 / 1000.0
            idx += 1
            vec[idx] = float(entry.get("p1_hp_delta", 0))
            idx += 1
            vec[idx] = float(entry.get("p2_hp_delta", 0))
            idx += 1
            vec[idx] = 1.0 if entry.get("was_fallback", False) else 0.0
            idx += 1

        return idx
