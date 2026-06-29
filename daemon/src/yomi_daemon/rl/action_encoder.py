"""Action encoder: maps between daemon action payloads and discrete action indices."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class ActionEncoder:
    """Encode/decode actions for RL.

    Actions are keyed by ``(action_name, canonical_payload)``. For actions with
    countable payloads (e.g. ``ParryHigh`` timing), a few discrete buckets are
    used so the action space stays small. Unknown payload shapes are preserved
    via a stable JSON canonicalization.
    """

    # Payload parameters that should be bucketed to keep the space small.
    COUNT_BUCKETS = [1, 2, 3, 5, 10, 20]

    def __init__(self, actions: Sequence[str | Mapping[str, Any]] | None = None) -> None:
        """Build the encoder.

        Args:
            actions: Sequence of action names (strings) or legal action dicts
                with an ``action`` key. Used to pre-populate the vocabulary.
        """
        self._action_to_index: dict[str, int] = {}
        self._index_to_action: dict[int, str] = {}
        self._index_to_payload: dict[int, Mapping[str, Any]] = {}

        if actions:
            for action in actions:
                if isinstance(action, str):
                    self._register(action)
                else:
                    self._register_from_legal_action(action)

    def _register(self, action_name: str, payload: Mapping[str, Any] | None = None) -> int:
        """Register an action and return its index."""
        key = self._canonical_key(action_name, payload)
        if key not in self._action_to_index:
            idx = len(self._action_to_index)
            self._action_to_index[key] = idx
            self._index_to_action[idx] = action_name
            self._index_to_payload[idx] = dict(payload) if payload else {}
        return self._action_to_index[key]

    def _register_from_legal_action(self, legal_action: Mapping[str, Any]) -> int:
        action_name = str(legal_action.get("action", ""))
        payload_spec = legal_action.get("payload_spec", {})
        default_payload = self._default_payload_from_spec(payload_spec)
        return self._register(action_name, default_payload)

    def _canonical_key(self, action_name: str, payload: Mapping[str, Any] | None) -> str:
        if not payload:
            return action_name
        # Bucket countable values to reduce the action space.
        bucketed: dict[str, Any] = {}
        for key, value in sorted(payload.items()):
            bucketed[key] = self._bucket_value(value)
        return f"{action_name}:{json.dumps(bucketed, sort_keys=True, separators=(',', ':'))}"

    def _bucket_value(self, value: Any) -> Any:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            for bucket in self.COUNT_BUCKETS:
                if value <= bucket:
                    return bucket
            return self.COUNT_BUCKETS[-1]
        if isinstance(value, float):
            return round(value, 2)
        if isinstance(value, Mapping):
            return {str(k): self._bucket_value(v) for k, v in sorted(value.items())}
        return value

    def _default_payload_from_spec(self, payload_spec: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """Extract a default payload from the JSON schema spec, if any."""
        properties = payload_spec.get("properties", {}) if isinstance(payload_spec, Mapping) else {}
        if not properties:
            return None

        defaults: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            if not isinstance(prop_schema, Mapping):
                continue
            inner = prop_schema
            # Handle nested object with a single countable value (e.g. "Melee Parry Timing")
            if inner.get("type") == "object" and "properties" in inner:
                sub_props = inner["properties"]
                if isinstance(sub_props, Mapping) and len(sub_props) == 1:
                    sub_name, sub_schema = next(iter(sub_props.items()))
                    if isinstance(sub_schema, Mapping):
                        default = sub_schema.get("default", 1)
                        defaults[prop_name] = {sub_name: default}
                        continue
            default = prop_schema.get("default")
            if default is not None:
                defaults[prop_name] = default

        return defaults if defaults else None

    def encode_action(
        self,
        action_name: str,
        data: Mapping[str, Any] | None = None,
    ) -> int:
        """Return the index for an action, registering it if necessary."""
        payload = self._normalize_payload(data)
        return self._register(action_name, payload)

    def encode_legal_actions(self, legal_actions: Sequence[Mapping[str, Any]]) -> list[int]:
        """Register all legal actions and return their indices."""
        indices: list[int] = []
        for legal_action in legal_actions:
            action_name = str(legal_action.get("action", ""))
            payload_spec = legal_action.get("payload_spec", {})
            default_payload = self._default_payload_from_spec(payload_spec)
            idx = self._register(action_name, default_payload)
            indices.append(idx)
        return indices

    def decode_action(self, index: int) -> tuple[str, Mapping[str, Any]]:
        """Return (action_name, payload) for an index."""
        if index not in self._index_to_action:
            raise IndexError(f"Action index {index} not in vocabulary")
        return self._index_to_action[index], dict(self._index_to_payload.get(index, {}))

    def build_mask(self, legal_action_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (mask, indices_array) for a set of legal action indices."""
        mask = np.zeros(self.vocab_size, dtype=np.float32)
        indices = np.array(legal_action_indices, dtype=np.int64)
        mask[indices] = 1.0
        return mask, indices

    def _normalize_payload(self, data: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not data:
            return None
        normalized: dict[str, Any] = {}
        for key, value in sorted(data.items()):
            normalized[key] = self._bucket_value(value)
        return normalized if normalized else None

    @property
    def vocab_size(self) -> int:
        return len(self._action_to_index)

    @property
    def action_to_index(self) -> dict[str, int]:
        return dict(self._action_to_index)
