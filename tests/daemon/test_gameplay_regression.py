"""Gameplay-quality regression suite for parameterized moves, history, and simultaneous turns (WU-022).

Protects the expanded gameplay surface with scenario coverage that reflects
how the real game stresses the bridge: parameterized payloads, prediction,
simultaneous-action windows, history-aware prompts, and multi-character matches.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import uuid

import pytest
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from websockets.asyncio.client import connect

from tests.daemon._decision_fixtures import (
    build_action,
    build_observation,
    build_request,
)
from yomi_daemon.adapters import build_policy_registry
from yomi_daemon.config import (
    DaemonRuntimeConfig,
    PolicyConfig,
    TournamentDefaults,
    TransportConfig,
    parse_runtime_config_document,
)
from yomi_daemon.prompt import render_prompt
from yomi_daemon.protocol import (
    CURRENT_PROTOCOL_VERSION,
    CURRENT_SCHEMA_VERSION,
    ActionDecision,
    CharacterSelectionConfig,
    CharacterSelectionMode,
    DecisionType,
    FallbackMode,
    FighterObservation,
    HistoryEntry,
    LegalAction,
    LoggingConfig,
    MessageType,
    Observation,
    PlayerPolicyMapping,
    Vector2,
)
from yomi_daemon.server import DaemonServer
from yomi_daemon.storage.writer import RUNS_DIR
from yomi_daemon.validation import (
    parse_envelope,
    validate_action_decision_for_request,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RUNTIME_DOC: dict[str, object] = {
    "version": "v1",
    "trace_seed": 22,
    "policy_mapping": {
        "p1": "baseline/random",
        "p2": "baseline/block_always",
    },
    "policies": {
        "baseline/random": {"provider": "baseline", "prompt_version": "none"},
        "baseline/block_always": {"provider": "baseline", "prompt_version": "none"},
        "baseline/greedy_damage": {"provider": "baseline", "prompt_version": "none"},
        "baseline/scripted_safe": {"provider": "baseline", "prompt_version": "none"},
    },
}

ALL_BASELINE_IDS = (
    "baseline/random",
    "baseline/block_always",
    "baseline/greedy_damage",
    "baseline/scripted_safe",
)


def _decide(policy_id: str, request: Any) -> ActionDecision:
    config = parse_runtime_config_document(_RUNTIME_DOC)
    registry = build_policy_registry(config)
    return asyncio.run(registry[policy_id].decide(request))


# -- Parameterized legal-action builders for each in-scope character ----------


def _cowboy_gun_toss() -> LegalAction:
    return build_action(
        "gun_toss",
        label="Gun Toss",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["distance", "target"],
            "properties": {
                "distance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 3,
                    "semantic_hint": "shot_distance",
                },
                "target": {
                    "type": "string",
                    "enum": ["enemy", "self"],
                    "default": "enemy",
                    "semantic_hint": "shot_target",
                },
            },
        },
        prediction=True,
        prediction_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["horizon"],
            "properties": {
                "horizon": {"type": "integer", "minimum": 1, "maximum": 3},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
        },
        damage=60.0,
        startup_frames=7,
    )


def _robot_drive_impact() -> LegalAction:
    return build_action(
        "drive_impact",
        label="Drive Impact",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["armor", "direction"],
            "properties": {
                "armor": {
                    "type": "boolean",
                    "default": False,
                    "semantic_hint": "spend_armor",
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "back", "up_forward", "down_forward"],
                    "default": "forward",
                    "semantic_hint": "drive_direction",
                },
            },
        },
        reverse=True,
        damage=90.0,
        startup_frames=12,
    )


def _ninja_sticky_bomb() -> LegalAction:
    return build_action(
        "sticky_bomb",
        label="Sticky Bomb",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["target_point"],
            "properties": {
                "target_point": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["x", "y"],
                    "properties": {
                        "x": {
                            "type": "number",
                            "minimum": -8.0,
                            "maximum": 8.0,
                        },
                        "y": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 6.0,
                        },
                    },
                },
            },
        },
        feint=True,
        damage=45.0,
        startup_frames=10,
    )


def _mutant_install_choice() -> LegalAction:
    return build_action(
        "install_choice",
        label="Install Choice",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["stance", "hold_position"],
            "properties": {
                "stance": {
                    "type": "string",
                    "enum": ["rush", "grasp", "slam"],
                    "default": "rush",
                    "semantic_hint": "install_stance",
                },
                "hold_position": {
                    "type": "boolean",
                    "default": False,
                    "semantic_hint": "preserve_spacing",
                },
            },
        },
        damage=0.0,
        startup_frames=15,
        meter_cost=1,
    )


def _wizard_conjure_storm() -> LegalAction:
    return build_action(
        "conjure_storm",
        label="Conjure Storm",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["charge_frames", "element", "direction"],
            "properties": {
                "charge_frames": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 20,
                    "default": 8,
                    "semantic_hint": "storm_charge",
                },
                "element": {
                    "type": "string",
                    "enum": ["wind", "fire", "lightning"],
                    "default": "wind",
                    "semantic_hint": "storm_element",
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "up_forward", "down_forward"],
                    "default": "forward",
                    "semantic_hint": "storm_direction",
                },
            },
        },
        damage=70.0,
        startup_frames=14,
        meter_cost=1,
    )


def _simple_block() -> LegalAction:
    return build_action("block", label="Block", description="Block safely.")


def _simple_jab() -> LegalAction:
    return build_action("jab", label="Jab", damage=20.0, startup_frames=3, di=True)


# -- Observation builders for specific characters ----------------------------


def _cowboy_vs_robot_observation(*, turn_id: int = 1) -> Observation:
    return Observation(
        tick=600 + turn_id,
        frame=12,
        active_player="p1",
        fighters=(
            FighterObservation(
                id="p1",
                character="Cowboy",
                hp=900,
                max_hp=1000,
                meter=1,
                burst=2,
                position=Vector2(x=-220.0, y=0.0),
                velocity=Vector2(x=0.0, y=0.0),
                facing="right",
                current_state="Idle",
                combo_count=0,
                blockstun=0,
                hitlag=0,
                state_interruptable=True,
                can_feint=True,
                grounded=True,
                character_data={"bullets_left": 4, "has_gun": True},
            ),
            FighterObservation(
                id="p2",
                character="Robot",
                hp=780,
                max_hp=1000,
                meter=0,
                burst=1,
                position=Vector2(x=220.0, y=0.0),
                velocity=Vector2(x=0.0, y=0.0),
                facing="left",
                current_state="Guard",
                combo_count=0,
                blockstun=0,
                hitlag=0,
                state_interruptable=True,
                can_feint=False,
                grounded=True,
            ),
        ),
        objects=(),
        stage={"id": "training_room"},
        history=(),
    )


def _wizard_vs_ninja_observation(*, turn_id: int = 1) -> Observation:
    return Observation(
        tick=604 + turn_id,
        frame=12,
        active_player="p1",
        fighters=(
            FighterObservation(
                id="p1",
                character="Wizard",
                hp=860,
                max_hp=1000,
                meter=1,
                burst=2,
                position=Vector2(x=-90.0, y=0.0),
                velocity=Vector2(x=0.0, y=0.0),
                facing="right",
                current_state="Hover",
                combo_count=0,
                blockstun=0,
                hitlag=0,
                state_interruptable=True,
                can_feint=True,
                grounded=True,
            ),
            FighterObservation(
                id="p2",
                character="Ninja",
                hp=790,
                max_hp=1000,
                meter=0,
                burst=1,
                position=Vector2(x=195.0, y=0.0),
                velocity=Vector2(x=0.0, y=0.0),
                facing="left",
                current_state="Idle",
                combo_count=0,
                blockstun=0,
                hitlag=0,
                state_interruptable=True,
                can_feint=True,
                grounded=True,
            ),
        ),
        objects=(),
        stage={"id": "training_room"},
        history=(),
    )


# -- Server-level helpers ---------------------------------------------------


def _unique_match_id() -> str:
    return f"regr-{uuid.uuid4().hex[:12]}"


def _server_runtime_config(
    *,
    p1: str = "baseline/random",
    p2: str = "baseline/block_always",
) -> DaemonRuntimeConfig:
    policies = {
        "baseline/random": PolicyConfig(provider="baseline"),
        "baseline/block_always": PolicyConfig(provider="baseline"),
        "baseline/greedy_damage": PolicyConfig(provider="baseline"),
        "baseline/scripted_safe": PolicyConfig(provider="baseline"),
    }
    return DaemonRuntimeConfig(
        version="v1",
        transport=TransportConfig(host="127.0.0.1", port=0),
        decision_timeout_ms=2500,
        fallback_mode=FallbackMode.SAFE_CONTINUE,
        logging=LoggingConfig(events=True, prompts=False, raw_provider_payloads=False),
        policy_mapping=PlayerPolicyMapping(p1=p1, p2=p2),
        policies=policies,
        character_selection=CharacterSelectionConfig(
            mode=CharacterSelectionMode.MIRROR,
        ),
        tournament=TournamentDefaults(
            format="round_robin",
            mirror_matches_first=True,
            side_swap=True,
            games_per_pair=10,
            fixed_stage="training_room",
        ),
        trace_seed=22,
    )


def _hello_envelope() -> dict[str, object]:
    return {
        "type": MessageType.HELLO.value,
        "version": CURRENT_PROTOCOL_VERSION.value,
        "ts": "2026-03-13T00:00:00Z",
        "payload": {
            "game_version": "1.9.20-steam",
            "mod_version": "0.0.1",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "supported_protocol_versions": [CURRENT_PROTOCOL_VERSION.value],
        },
    }


def _match_ended_envelope(*, match_id: str, total_turns: int) -> dict[str, Any]:
    return {
        "type": MessageType.MATCH_ENDED.value,
        "version": CURRENT_PROTOCOL_VERSION.value,
        "ts": "2026-03-13T00:01:00Z",
        "payload": {
            "match_id": match_id,
            "winner": "p1",
            "end_reason": "ko",
            "total_turns": total_turns,
            "end_tick": 900,
            "end_frame": 60,
            "errors": [],
        },
    }


def _fighter_dict(
    player_id: str,
    character: str,
    *,
    hp: int = 1000,
    position_x: float = 0.0,
    facing: str = "right",
    current_state: str = "neutral",
    can_feint: bool = False,
    character_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "id": player_id,
        "character": character,
        "hp": hp,
        "max_hp": 1000,
        "meter": 1,
        "burst": 1,
        "position": {"x": position_x, "y": 0.0},
        "velocity": {"x": 0.0, "y": 0.0},
        "facing": facing,
        "current_state": current_state,
        "combo_count": 0,
        "blockstun": 0,
        "hitlag": 0,
        "state_interruptable": True,
        "can_feint": can_feint,
        "grounded": True,
    }
    if character_data:
        d["character_data"] = character_data
    return d


def _action_dict(
    action: str,
    *,
    label: str | None = None,
    payload_spec: dict[str, Any] | None = None,
    di: bool = False,
    feint: bool = False,
    reverse: bool = False,
    prediction: bool = False,
    prediction_spec: dict[str, Any] | None = None,
    damage: float | None = None,
    startup_frames: int | None = None,
) -> dict[str, Any]:
    d: dict[str, Any] = {
        "action": action,
        "payload_spec": payload_spec or {},
        "supports": {
            "di": di,
            "feint": feint,
            "reverse": reverse,
            "prediction": prediction,
        },
    }
    if label:
        d["label"] = label
    if prediction_spec:
        d["prediction_spec"] = prediction_spec
    if damage is not None:
        d["damage"] = damage
    if startup_frames is not None:
        d["startup_frames"] = startup_frames
    return d


def _decision_request_envelope(
    *,
    match_id: str,
    turn_id: int,
    player_id: str,
    observation: dict[str, Any],
    legal_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": MessageType.DECISION_REQUEST.value,
        "version": CURRENT_PROTOCOL_VERSION.value,
        "ts": "2026-03-13T00:00:01Z",
        "payload": {
            "match_id": match_id,
            "turn_id": turn_id,
            "player_id": player_id,
            "deadline_ms": 2500,
            "state_hash": f"state-{turn_id}",
            "legal_actions_hash": f"legal-{turn_id}",
            "decision_type": DecisionType.TURN_ACTION.value,
            "observation": observation,
            "legal_actions": legal_actions,
        },
    }


def _observation_dict(
    turn_id: int,
    player_id: str,
    p1_character: str,
    p2_character: str,
    *,
    history: list[dict[str, Any]] | None = None,
    p1_hp: int = 1000,
    p2_hp: int = 1000,
) -> dict[str, Any]:
    return {
        "tick": 100 + turn_id,
        "frame": 12,
        "active_player": player_id,
        "fighters": [
            _fighter_dict(
                "p1",
                p1_character,
                hp=p1_hp,
                position_x=-200.0,
                facing="right",
                can_feint=True,
            ),
            _fighter_dict(
                "p2",
                p2_character,
                hp=p2_hp,
                position_x=200.0,
                facing="left",
            ),
        ],
        "objects": [],
        "stage": {"id": "training_room"},
        "history": history or [],
    }


def _find_match_dir(match_id: str) -> Path | None:
    dirs = list(RUNS_DIR.glob(f"*{match_id}"))
    return dirs[0] if len(dirs) == 1 else None


# ---------------------------------------------------------------------------
# 1. Simultaneous-action windows and dual-player actionable turns
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_simultaneous_turns_both_players_receive_valid_decisions() -> None:
    """When both players are actionable on the same tick, the server handles
    interleaved decision requests for p1 and p2 correctly."""

    mid = _unique_match_id()
    config = _server_runtime_config()
    match_dir: Path | None = None

    async def scenario() -> list[ActionDecision]:
        server = DaemonServer(
            port=0,
            policy_mapping=config.policy_mapping,
            config_snapshot=config.to_config_payload(),
            runtime_config=config,
        )
        await server.start()
        decisions: list[ActionDecision] = []
        try:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await ws.send(json.dumps(_hello_envelope()))
                await ws.recv()  # hello_ack

                # Simulate 3 simultaneous-turn windows: both players act on the
                # same turn_id, just like the real game when both become actionable.
                for turn in range(1, 4):
                    obs = _observation_dict(turn, "p1", "Ninja", "Wizard")
                    actions_p1 = [
                        _action_dict("jab", label="Jab", damage=20.0, di=True),
                        _action_dict("block", label="Block"),
                    ]
                    actions_p2 = [
                        _action_dict("fireball", label="Fireball", damage=40.0),
                        _action_dict("block", label="Block"),
                    ]

                    # p1 request
                    req_p1 = _decision_request_envelope(
                        match_id=mid,
                        turn_id=turn * 2 - 1,
                        player_id="p1",
                        observation=obs,
                        legal_actions=actions_p1,
                    )
                    await ws.send(json.dumps(req_p1))
                    raw = await ws.recv()
                    resp = parse_envelope(json.loads(raw))
                    assert resp.type is MessageType.ACTION_DECISION
                    decisions.append(cast(ActionDecision, resp.payload))

                    # p2 request on same tick
                    obs_p2 = _observation_dict(turn, "p2", "Ninja", "Wizard")
                    req_p2 = _decision_request_envelope(
                        match_id=mid,
                        turn_id=turn * 2,
                        player_id="p2",
                        observation=obs_p2,
                        legal_actions=actions_p2,
                    )
                    await ws.send(json.dumps(req_p2))
                    raw = await ws.recv()
                    resp = parse_envelope(json.loads(raw))
                    assert resp.type is MessageType.ACTION_DECISION
                    decisions.append(cast(ActionDecision, resp.payload))

                await ws.send(
                    json.dumps(_match_ended_envelope(match_id=mid, total_turns=6))
                )
        finally:
            await server.stop()
        return decisions

    try:
        decisions = asyncio.run(scenario())
        assert len(decisions) == 6

        # p1 decisions (odd indices in the sequence)
        for i in range(0, 6, 2):
            assert decisions[i].action in {"jab", "block"}

        # p2 decisions (even indices)
        for i in range(1, 6, 2):
            assert decisions[i].action in {"fireball", "block"}

    finally:
        match_dir = _find_match_dir(mid)
        if match_dir is not None and match_dir.exists():
            shutil.rmtree(match_dir)


def test_simultaneous_turns_baseline_adapters_handle_dual_requests() -> None:
    """All baseline adapters produce valid decisions when given requests that
    simulate simultaneous-turn windows (same observation, different players)."""

    actions = (
        _simple_block(),
        _simple_jab(),
        _cowboy_gun_toss(),
    )

    for policy_id in ALL_BASELINE_IDS:
        # p1 request
        request_p1 = build_request(actions, match_id="simul-001", turn_id=1)
        decision_p1 = _decide(policy_id, request_p1)
        assert decision_p1.action in {"block", "jab", "gun_toss"}

        # p2 request on the same logical tick
        obs_p2 = replace(build_observation(), active_player="p2")
        request_p2 = replace(
            build_request(actions, match_id="simul-001", turn_id=2),
            player_id="p2",
            observation=obs_p2,
        )
        decision_p2 = _decide(policy_id, request_p2)
        assert decision_p2.action in {"block", "jab", "gun_toss"}


# ---------------------------------------------------------------------------
# 2. Full-match regression with multiple characters and parameterized actions
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_full_match_cowboy_vs_wizard_with_parameterized_actions() -> None:
    """Run a multi-turn server match alternating Cowboy and Wizard parameterized
    actions alongside simple moves, proving the richer contract works end to end."""

    mid = _unique_match_id()
    config = _server_runtime_config(
        p1="baseline/greedy_damage",
        p2="baseline/scripted_safe",
    )
    match_dir: Path | None = None

    # Cowboy parameterized action (gun toss with slider + option + prediction)
    cowboy_gun_toss_dict = _action_dict(
        "gun_toss",
        label="Gun Toss",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["distance", "target"],
            "properties": {
                "distance": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 3,
                },
                "target": {
                    "type": "string",
                    "enum": ["enemy", "self"],
                    "default": "enemy",
                },
            },
        },
        prediction=True,
        prediction_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["horizon"],
            "properties": {
                "horizon": {"type": "integer", "minimum": 1, "maximum": 3},
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
            },
        },
        damage=60.0,
        startup_frames=7,
    )

    # Wizard parameterized action (slider + option + 8way)
    wizard_storm_dict = _action_dict(
        "conjure_storm",
        label="Conjure Storm",
        payload_spec={
            "type": "object",
            "additionalProperties": False,
            "required": ["charge_frames", "element", "direction"],
            "properties": {
                "charge_frames": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 20,
                    "default": 8,
                },
                "element": {
                    "type": "string",
                    "enum": ["wind", "fire", "lightning"],
                    "default": "wind",
                },
                "direction": {
                    "type": "string",
                    "enum": ["forward", "up_forward", "down_forward"],
                    "default": "forward",
                },
            },
        },
        damage=70.0,
        startup_frames=14,
    )

    simple_actions = [
        _action_dict("block", label="Block"),
        _action_dict("jab", label="Jab", damage=20.0, startup_frames=3, di=True),
    ]

    TOTAL_TURNS = 8
    history: list[dict[str, Any]] = []

    async def scenario() -> list[ActionDecision]:
        server = DaemonServer(
            port=0,
            policy_mapping=config.policy_mapping,
            config_snapshot=config.to_config_payload(),
            runtime_config=config,
        )
        await server.start()
        decisions: list[ActionDecision] = []
        try:
            async with connect(f"ws://127.0.0.1:{server.listening_port}") as ws:
                await ws.send(json.dumps(_hello_envelope()))
                await ws.recv()

                for turn in range(1, TOTAL_TURNS + 1):
                    player = "p1" if turn % 2 == 1 else "p2"

                    # Alternate characters: odd turns Cowboy(p1) vs Wizard(p2),
                    # even turns Wizard(p2) acts.
                    if player == "p1":
                        actions = simple_actions + [cowboy_gun_toss_dict]
                    else:
                        actions = simple_actions + [wizard_storm_dict]

                    obs = _observation_dict(
                        turn,
                        player,
                        "Cowboy",
                        "Wizard",
                        history=history[-10:],  # bounded sliding window
                        p1_hp=max(1000 - turn * 30, 100),
                        p2_hp=max(1000 - turn * 25, 100),
                    )

                    req = _decision_request_envelope(
                        match_id=mid,
                        turn_id=turn,
                        player_id=player,
                        observation=obs,
                        legal_actions=actions,
                    )
                    await ws.send(json.dumps(req))
                    raw = await ws.recv()
                    resp = parse_envelope(json.loads(raw))
                    assert resp.type is MessageType.ACTION_DECISION
                    d = cast(ActionDecision, resp.payload)
                    decisions.append(d)

                    # Accumulate history for subsequent turns
                    history.append(
                        {
                            "turn_id": turn,
                            "player_id": player,
                            "action": d.action,
                            "was_fallback": d.fallback_reason is not None,
                        }
                    )

                await ws.send(
                    json.dumps(
                        _match_ended_envelope(match_id=mid, total_turns=TOTAL_TURNS)
                    )
                )
        finally:
            await server.stop()
        return decisions

    try:
        decisions = asyncio.run(scenario())
        assert len(decisions) == TOTAL_TURNS

        valid_p1 = {"block", "jab", "gun_toss"}
        valid_p2 = {"block", "jab", "conjure_storm"}

        for i, d in enumerate(decisions):
            turn = i + 1
            if turn % 2 == 1:
                assert d.action in valid_p1, (
                    f"Turn {turn}: {d.action} not in {valid_p1}"
                )
            else:
                assert d.action in valid_p2, (
                    f"Turn {turn}: {d.action} not in {valid_p2}"
                )

        # No fallbacks should occur
        fallbacks = [d for d in decisions if d.fallback_reason is not None]
        assert len(fallbacks) == 0, (
            f"Unexpected fallbacks: {[d.action for d in fallbacks]}"
        )

    finally:
        match_dir = _find_match_dir(mid)
        if match_dir is not None and match_dir.exists():
            shutil.rmtree(match_dir)


# ---------------------------------------------------------------------------
# 3. Fallback-rate and legality-rate assertions for parameterized fixtures
# ---------------------------------------------------------------------------


def test_all_baselines_zero_fallback_on_cowboy_parameterized() -> None:
    """All baseline adapters produce valid, non-fallback decisions for Cowboy gun toss."""
    actions = (_simple_block(), _simple_jab(), _cowboy_gun_toss())
    request = replace(
        build_request(actions, match_id="fb-cowboy"),
        observation=_cowboy_vs_robot_observation(),
    )
    for policy_id in ALL_BASELINE_IDS:
        decision = _decide(policy_id, request)
        assert decision.fallback_reason is None, (
            f"{policy_id} fell back: {decision.fallback_reason}"
        )
        validate_action_decision_for_request(request, decision)


def test_aggregate_fallback_rate_across_all_parameterized_fixtures() -> None:
    """Aggregate test: run every baseline against every parameterized character fixture.
    Assert an overall fallback rate of zero."""

    parameterized_actions = [
        ("cowboy", _cowboy_gun_toss()),
        ("robot", _robot_drive_impact()),
        ("ninja", _ninja_sticky_bomb()),
        ("mutant", _mutant_install_choice()),
        ("wizard", _wizard_conjure_storm()),
    ]

    total = 0
    fallbacks = 0

    for char_name, param_action in parameterized_actions:
        actions = (_simple_block(), _simple_jab(), param_action)
        request = build_request(actions, match_id=f"agg-{char_name}")
        for policy_id in ALL_BASELINE_IDS:
            decision = _decide(policy_id, request)
            total += 1
            if decision.fallback_reason is not None:
                fallbacks += 1

    assert fallbacks == 0, (
        f"Fallback rate: {fallbacks}/{total} ({100 * fallbacks / total:.1f}%)"
    )
    assert total == 20  # 5 characters * 4 baselines


def test_parameterized_decisions_pass_legality_validation() -> None:
    """Decisions from all baselines for all parameterized actions pass full
    request-relative legality validation."""

    parameterized_actions = [
        _cowboy_gun_toss(),
        _robot_drive_impact(),
        _ninja_sticky_bomb(),
        _mutant_install_choice(),
        _wizard_conjure_storm(),
    ]

    for param_action in parameterized_actions:
        actions = (_simple_block(), param_action)
        request = build_request(actions, match_id="legality-check")
        for policy_id in ALL_BASELINE_IDS:
            decision = _decide(policy_id, request)
            # This will raise DecisionValidationError if the decision is illegal
            matched = validate_action_decision_for_request(request, decision)
            assert matched.action == decision.action


# ---------------------------------------------------------------------------
# 4. History-aware prompt rendering regression tests
# ---------------------------------------------------------------------------


def _build_history(n: int) -> tuple[HistoryEntry, ...]:
    """Build a realistic history of n entries alternating players.

    Uses enriched per-player fields because the compact observation renderer
    keeps p1_action/p2_action and related fields, not the legacy single-player
    player_id/action fields.
    """
    entries: list[HistoryEntry] = []
    actions_cycle = ["block", "jab", "gun_toss", "block", "conjure_storm"]
    for i in range(n):
        is_p1 = i % 2 == 0
        action = actions_cycle[i % len(actions_cycle)]
        entries.append(
            HistoryEntry(
                turn_id=i + 1,
                p1_action=action if is_p1 else None,
                p2_action=None if is_p1 else action,
                p1_was_fallback=(i == 3 and is_p1),
                p2_was_fallback=(i == 3 and not is_p1),
            )
        )
    return tuple(entries)


def test_prompt_renders_history_entries() -> None:
    """Prompt text must include history entries from the observation."""
    history = _build_history(5)
    obs = replace(build_observation(), history=history)
    request = replace(
        build_request((_simple_block(), _cowboy_gun_toss())),
        observation=obs,
    )

    rendered = render_prompt(request, configured_prompt_version="minimal_v1")

    # History entries should appear in the compact observation JSON.
    assert '"history"' in rendered.prompt_text
    assert '"gun_toss"' in rendered.prompt_text
    assert '"conjure_storm"' in rendered.prompt_text
    # i == 3 is a p2 turn, so the fallback flag lands on p2.
    assert '"p2_was_fallback": true' in rendered.prompt_text


def test_prompt_history_interacts_with_parameterized_actions() -> None:
    """A prompt with both history and parameterized legal actions renders both correctly."""
    history = _build_history(3)
    obs = replace(
        _wizard_vs_ninja_observation(),
        history=history,
    )
    request = replace(
        build_request(
            (_simple_block(), _wizard_conjure_storm(), _ninja_sticky_bomb()),
        ),
        observation=obs,
    )

    rendered = render_prompt(request, configured_prompt_version="minimal_v1")

    # History
    assert '"history"' in rendered.prompt_text
    assert '"turn_id": 1' in rendered.prompt_text

    # Parameterized action specs
    assert '"storm_charge"' in rendered.prompt_text
    assert '"sticky_bomb"' in rendered.prompt_text
    assert '"target_point"' in rendered.prompt_text
