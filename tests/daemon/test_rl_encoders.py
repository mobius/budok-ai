"""Tests for RL observation/action encoders and trajectory extraction."""

from __future__ import annotations

import numpy as np
import pytest

from yomi_daemon.rl import ObservationEncoder, ActionEncoder


def _sample_observation() -> dict[str, object]:
    return {
        "tick": 10,
        "frame": 9,
        "active_player": "p1",
        "fighters": [
            {
                "id": "p1",
                "character": "Cowboy",
                "hp": 150,
                "max_hp": 200,
                "meter": 25,
                "burst": 1,
                "position": {"x": -30.0, "y": 0.0},
                "velocity": {"x": 1.0, "y": 0.0},
                "facing": "right",
                "current_state": "Idle",
                "combo_count": 0,
                "blockstun": 0,
                "hitlag": 0,
                "state_interruptable": True,
                "can_feint": False,
                "grounded": True,
                "initiative": True,
                "combo_proration": 0.0,
                "character_data": {"bullets_left": 5, "consecutive_shots": 1, "has_gun": True},
            },
            {
                "id": "p2",
                "character": "Wizard",
                "hp": 180,
                "max_hp": 200,
                "meter": 10,
                "burst": 1,
                "position": {"x": 40.0, "y": 0.0},
                "velocity": {"x": -1.0, "y": 0.0},
                "facing": "left",
                "current_state": "Start",
                "combo_count": 0,
                "blockstun": 0,
                "hitlag": 0,
                "state_interruptable": True,
                "can_feint": False,
                "grounded": True,
                "initiative": True,
                "combo_proration": 0.0,
                "character_data": {"geyser_charge": 0, "gusts_in_combo": 0, "hover_left": 900},
            },
        ],
        "objects": [],
        "stage": {"ceiling_height": 400, "has_ceiling": True, "width": 1100},
        "history": [
            {
                "turn_id": 1,
                "p1_action": "DashForward",
                "p2_action": "ParryHigh",
                "p1_hp_delta": 0,
                "p2_hp_delta": 0,
                "was_fallback": False,
            }
        ],
    }


def test_observation_encoder_returns_fixed_size_vector() -> None:
    encoder = ObservationEncoder(history_len=3)
    vec = encoder.encode(_sample_observation())

    assert isinstance(vec, np.ndarray)
    assert vec.dtype == np.float32
    assert vec.shape == (encoder.vector_size,)
    assert encoder.vector_size == 63


def test_observation_encoder_active_player_encoded() -> None:
    encoder = ObservationEncoder()
    vec = encoder.encode(_sample_observation())
    assert vec[0] == 0.0  # p1

    obs = dict(_sample_observation())
    obs["active_player"] = "p2"
    vec2 = encoder.encode(obs)
    assert vec2[0] == 1.0


def test_action_encoder_registers_and_decodes_actions() -> None:
    encoder = ActionEncoder()
    idx = encoder.encode_action("HSlash2")
    assert idx == 0

    idx2 = encoder.encode_action("ParryHigh", {"Melee Parry Timing": {"count": 3}})
    assert idx2 == 1

    name, payload = encoder.decode_action(idx2)
    assert name == "ParryHigh"
    assert payload == {"Melee Parry Timing": {"count": 3}}


def test_action_encoder_builds_legal_mask() -> None:
    encoder = ActionEncoder()
    legal = [encoder.encode_action("HSlash2"), encoder.encode_action("DashForward")]
    mask, indices = encoder.build_mask(legal)

    assert mask.shape == (encoder.vocab_size,)
    assert np.all(mask[indices] == 1.0)
    assert np.sum(mask) == 2.0


def test_action_encoder_buckets_countable_payloads() -> None:
    encoder = ActionEncoder()
    idx1 = encoder.encode_action("ParryHigh", {"Melee Parry Timing": {"count": 4}})
    idx2 = encoder.encode_action("ParryHigh", {"Melee Parry Timing": {"count": 5}})
    # 4 and 5 both fall into the bucket of 5
    assert idx1 == idx2


def test_action_encoder_unknown_action_gets_registered() -> None:
    encoder = ActionEncoder()
    idx = encoder.encode_action("Teleport")
    name, payload = encoder.decode_action(idx)
    assert name == "Teleport"
    assert payload == {}
