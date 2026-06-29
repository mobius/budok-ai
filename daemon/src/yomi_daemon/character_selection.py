"""LLM-driven character selection for llm_choice mode."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from hashlib import sha256
from json import JSONDecodeError
from random import Random
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

from yomi_daemon.prompt import (
    VALID_CHARACTERS,
    MatchHistoryEntry,
    character_select_output_json_schema,
    render_character_select_prompt,
)
from yomi_daemon.protocol import CharacterAssignments, PlayerSlot

if TYPE_CHECKING:
    from yomi_daemon.config import DaemonRuntimeConfig, PolicyConfig


logger = logging.getLogger("yomi_daemon.character_selection")

_TOOL_NAME = "submit_character_choice"


@dataclass(frozen=True, slots=True)
class CharacterSelectionTrace:
    """Trace of a single player's character selection."""

    player_slot: str
    policy_id: str
    character: str
    reasoning: str
    prompt_text: str
    raw_response: object
    match_history: list[MatchHistoryEntry] = field(default_factory=list)
    fallback: bool = False

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "player_slot": self.player_slot,
            "policy_id": self.policy_id,
            "character": self.character,
            "reasoning": self.reasoning,
            "prompt_text": self.prompt_text,
            "fallback": self.fallback,
        }
        if self.match_history:
            result["match_history"] = [
                {
                    "your_character": e.your_character,
                    "opponent_character": e.opponent_character,
                    "result": e.result,
                    "your_final_hp": e.your_final_hp,
                    "opponent_final_hp": e.opponent_final_hp,
                }
                for e in self.match_history
            ]
        return result


@dataclass(frozen=True, slots=True)
class CharacterSelectionResult:
    """Result of character selection with traces for both players."""

    assignments: CharacterAssignments
    traces: list[CharacterSelectionTrace]


@dataclass(frozen=True, slots=True)
class _PlayerResult:
    character: str
    trace: CharacterSelectionTrace


async def resolve_character_assignments(
    config: "DaemonRuntimeConfig",
    *,
    match_history: dict[str, list[MatchHistoryEntry]] | None = None,
) -> CharacterSelectionResult:
    """Resolve character assignments for llm_choice mode.

    For provider-backed policies, calls the LLM with the character selection prompt.
    For baseline policies, picks a random character seeded by trace_seed.
    Both players are resolved concurrently.

    Returns a CharacterSelectionResult with assignments and reasoning traces.
    """
    p1_policy_id = config.policy_mapping.p1
    p2_policy_id = config.policy_mapping.p2
    p1_config = config.policies[p1_policy_id]
    p2_config = config.policies[p2_policy_id]

    history = match_history or {}

    p1_coro = _resolve_for_player(
        player_slot=PlayerSlot.P1,
        policy_id=p1_policy_id,
        policy_config=p1_config,
        opponent_policy_id=p2_policy_id,
        trace_seed=config.trace_seed,
        match_history=history.get("p1"),
        timeout_ms=config.decision_timeout_ms,
    )
    p2_coro = _resolve_for_player(
        player_slot=PlayerSlot.P2,
        policy_id=p2_policy_id,
        policy_config=p2_config,
        opponent_policy_id=p1_policy_id,
        trace_seed=config.trace_seed,
        match_history=history.get("p2"),
        timeout_ms=config.decision_timeout_ms,
    )

    p1_result, p2_result = await asyncio.gather(p1_coro, p2_coro)

    logger.info(
        "LLM character selection: P1 (%s) -> %s, P2 (%s) -> %s",
        p1_policy_id,
        p1_result.character,
        p2_policy_id,
        p2_result.character,
    )

    return CharacterSelectionResult(
        assignments=CharacterAssignments(p1=p1_result.character, p2=p2_result.character),
        traces=[p1_result.trace, p2_result.trace],
    )


async def _resolve_for_player(
    *,
    player_slot: PlayerSlot,
    policy_id: str,
    policy_config: "PolicyConfig",
    opponent_policy_id: str,
    trace_seed: int,
    match_history: list[MatchHistoryEntry] | None,
    timeout_ms: int,
) -> _PlayerResult:
    """Resolve a character for a single player."""
    if policy_config.provider == "baseline":
        char = _random_character(trace_seed=trace_seed, salt=f"char_select:{player_slot.value}")
        trace = CharacterSelectionTrace(
            player_slot=player_slot.value,
            policy_id=policy_id,
            character=char,
            reasoning="baseline random selection",
            prompt_text="",
            raw_response=None,
            fallback=False,
        )
        return _PlayerResult(character=char, trace=trace)

    try:
        return await _llm_character_select(
            player_slot=player_slot,
            policy_id=policy_id,
            policy_config=policy_config,
            opponent_policy_id=opponent_policy_id,
            match_history=match_history,
            timeout_ms=timeout_ms,
        )
    except Exception:
        logger.exception(
            "Character selection failed for %s (%s), falling back to random",
            player_slot.value,
            policy_id,
        )
        char = _random_character(trace_seed=trace_seed, salt=f"char_select:{player_slot.value}")
        trace = CharacterSelectionTrace(
            player_slot=player_slot.value,
            policy_id=policy_id,
            character=char,
            reasoning="fallback to random after LLM failure",
            prompt_text="",
            raw_response=None,
            fallback=True,
        )
        return _PlayerResult(character=char, trace=trace)


def _random_character(*, trace_seed: int, salt: str) -> str:
    """Pick a random character deterministically from trace_seed + salt."""
    seed_material = f"character_select|{trace_seed}|{salt}"
    digest = sha256(seed_material.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = Random(seed)
    return rng.choice(VALID_CHARACTERS)


async def _llm_character_select(
    *,
    player_slot: PlayerSlot,
    policy_id: str,
    policy_config: "PolicyConfig",
    opponent_policy_id: str,
    match_history: list[MatchHistoryEntry] | None,
    timeout_ms: int,
) -> _PlayerResult:
    """Call the LLM to choose a character."""
    prompt_text = render_character_select_prompt(
        player_id=player_slot.value,
        opponent_policy_id=opponent_policy_id,
        match_history=match_history,
    )

    provider = policy_config.provider
    if provider == "anthropic":
        raw = await _call_anthropic(
            policy_config=policy_config,
            prompt_text=prompt_text,
            timeout_ms=timeout_ms,
        )
    elif provider in ("openai", "openrouter"):
        raw = await _call_openai_compat(
            policy_config=policy_config,
            prompt_text=prompt_text,
            timeout_ms=timeout_ms,
            provider=provider,
        )
    else:
        raise ValueError(f"Unsupported provider for character selection: {provider}")

    character, reasoning = _parse_character_choice(raw)

    logger.info("Character selection reasoning: %s", reasoning)

    trace = CharacterSelectionTrace(
        player_slot=player_slot.value,
        policy_id=policy_id,
        character=character,
        reasoning=reasoning,
        prompt_text=prompt_text,
        raw_response=raw,
        match_history=list(match_history) if match_history else [],
        fallback=False,
    )
    return _PlayerResult(character=character, trace=trace)


def _parse_character_choice(raw: object) -> tuple[str, str]:
    """Extract and validate the character choice from provider response.

    Returns (character, reasoning) tuple.
    """
    if isinstance(raw, str):
        # Try to extract JSON from text
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON object in text
            text = cast(str, raw)
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    raw = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    pass

    if isinstance(raw, dict):
        raw_dict = cast(dict[str, object], raw)
        character = raw_dict.get("character")
        if isinstance(character, str) and character in VALID_CHARACTERS:
            reasoning = str(raw_dict.get("reasoning", ""))
            return character, reasoning

    raise ValueError(f"Could not parse valid character from response: {raw!r}")


def _extract_json_object(text: str) -> Mapping[str, object] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(parsed, Mapping):
            return cast(Mapping[str, object], parsed)
    return None


async def _call_anthropic(
    *,
    policy_config: "PolicyConfig",
    prompt_text: str,
    timeout_ms: int,
) -> object:
    """Make an Anthropic API call for character selection."""
    from anthropic import AsyncAnthropic

    api_key = policy_config.credential.value
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured for character selection")

    client = AsyncAnthropic(api_key=api_key)
    schema = character_select_output_json_schema()

    payload: dict[str, Any] = {
        "model": cast(str, policy_config.model),
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt_text}\n\nUse the `{_TOOL_NAME}` tool for your answer.",
                    }
                ],
            }
        ],
        "tools": [
            {
                "name": _TOOL_NAME,
                "description": "Submit your character choice with reasoning.",
                "input_schema": schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": _TOOL_NAME},
    }
    if policy_config.temperature is not None:
        payload["temperature"] = policy_config.temperature

    response = await client.messages.create(**payload, timeout=timeout_ms / 1000)
    dumped = response.model_dump(mode="json")

    # Extract tool_use output
    content = dumped.get("content", [])
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            if block.get("name") == _TOOL_NAME:
                return block.get("input", {})

    # Fall back to text
    text_parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(block.get("text", ""))
    if text_parts:
        return "".join(text_parts)

    raise ValueError("No usable output from Anthropic character selection response")


async def _call_openai_compat(
    *,
    policy_config: "PolicyConfig",
    prompt_text: str,
    timeout_ms: int,
    provider: str,
) -> object:
    """Make an OpenAI-compatible API call for character selection."""
    from openai import AsyncOpenAI

    api_key = policy_config.credential.value
    if not api_key:
        raise ValueError(f"API key not configured for {provider} character selection")

    options = dict(policy_config.options)
    custom_base_url = options.get("base_url")
    response_format = options.get("response_format", "json_schema")

    base_url: str | None = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
    if isinstance(custom_base_url, str) and custom_base_url:
        base_url = custom_base_url

    default_headers: dict[str, str] = {}
    if provider == "openrouter" and not isinstance(custom_base_url, str):
        default_headers["X-Title"] = "budok-ai"
        default_headers["X-OpenRouter-Categories"] = "game"
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers or None,
    )

    if response_format == "json_object":
        response_format_payload: dict[str, Any] = {"type": "json_object"}
    else:
        schema = character_select_output_json_schema()
        response_format_payload = {
            "type": "json_schema",
            "json_schema": {
                "name": "character_choice",
                "strict": True,
                "schema": schema,
            },
        }

    payload: dict[str, Any] = {
        "model": cast(str, policy_config.model),
        "max_tokens": 2048,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"{prompt_text}\n\nRespond with a JSON object containing "
                    f'"reasoning" (string) and "character" (one of: '
                    f"{', '.join(VALID_CHARACTERS)})."
                ),
            }
        ],
        "response_format": response_format_payload,
    }
    if policy_config.temperature is not None:
        payload["temperature"] = policy_config.temperature

    response = await client.chat.completions.create(**payload, timeout=timeout_ms / 1000)
    dumped = response.model_dump(mode="json")

    # Extract from chat completions response
    choices = dumped.get("choices", [])
    if not choices:
        raise ValueError(f"No choices in {provider} character selection response")

    message = choices[0].get("message", {})

    # Try parsed structured output first
    parsed = message.get("parsed")
    if isinstance(parsed, dict):
        return parsed

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                decoded = _extract_json_object(arguments)
                if decoded is not None:
                    return dict(decoded)

    # Fall back to content text
    content = message.get("content", "")
    if isinstance(content, str) and content.strip():
        return content

    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        decoded = _extract_json_object(reasoning)
        if decoded is not None:
            return dict(decoded)

    raise ValueError(f"No usable output from {provider} character selection response")
