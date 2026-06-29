"""Adapter contracts, metadata, and construction helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from yomi_daemon.protocol import (
    ActionDecision,
    DIVector,
    DecisionExtras,
    DecisionRequest,
    JsonObject,
    LegalAction,
    PlayerSlot,
)

if TYPE_CHECKING:
    from yomi_daemon.config import DaemonRuntimeConfig, PolicyConfig


class AdapterConstructionError(ValueError):
    """Raised when a configured policy cannot be constructed."""


@dataclass(frozen=True, slots=True)
class PolicyAdapterMetadata:
    policy_id: str
    provider: str
    model: str | None = None
    prompt_version: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PromptTrace:
    prompt_text: str
    prompt_version: str | None = None
    provider_request: JsonObject | None = None
    provider_response: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class PolicyDecisionResult:
    decision: ActionDecision
    prompt_trace: PromptTrace | None = None


@runtime_checkable
class PolicyAdapter(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def metadata(self) -> PolicyAdapterMetadata: ...

    async def decide(self, request: DecisionRequest) -> ActionDecision: ...

    async def decide_with_trace(self, request: DecisionRequest) -> PolicyDecisionResult: ...


class BasePolicyAdapter(ABC):
    """Common utilities for deterministic adapter implementations."""

    def __init__(
        self,
        *,
        metadata: PolicyAdapterMetadata,
        default_trace_seed: int = 0,
    ) -> None:
        self._metadata = PolicyAdapterMetadata(
            policy_id=metadata.policy_id,
            provider=metadata.provider,
            model=metadata.model,
            prompt_version=metadata.prompt_version,
            options=MappingProxyType(dict(metadata.options)),
        )
        self._default_trace_seed = default_trace_seed

    @property
    def id(self) -> str:
        return self._metadata.policy_id

    @property
    def metadata(self) -> PolicyAdapterMetadata:
        return self._metadata

    @property
    def default_trace_seed(self) -> int:
        return self._default_trace_seed

    def rng_for_request(self, request: DecisionRequest, *, salt: str = "") -> Random:
        seed = self.seed_for_request(request, salt=salt)
        return Random(seed)

    def seed_for_request(self, request: DecisionRequest, *, salt: str = "") -> int:
        trace_seed = (
            request.trace_seed if request.trace_seed is not None else self.default_trace_seed
        )
        seed_material = "|".join(
            (
                self.id,
                str(trace_seed),
                request.match_id,
                request.player_id,
                str(request.turn_id),
                request.state_hash,
                request.legal_actions_hash,
                salt,
            )
        )
        digest = sha256(seed_material.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=False)

    def build_decision(
        self,
        request: DecisionRequest,
        action: LegalAction,
        *,
        data: JsonObject | None,
        notes: str | None = None,
    ) -> ActionDecision:
        return ActionDecision(
            match_id=request.match_id,
            turn_id=request.turn_id,
            action=action.action,
            data=data,
            extra=self.default_extras(action),
            policy_id=self.id,
            notes=notes,
        )

    def default_extras(self, action: LegalAction) -> DecisionExtras:
        return DecisionExtras(
            di=DIVector(x=0, y=0) if action.supports.di else None,
            feint=False,
            reverse=False,
            prediction=None,
        )

    async def decide_with_trace(self, request: DecisionRequest) -> PolicyDecisionResult:
        return PolicyDecisionResult(decision=await self.decide(request))

    @abstractmethod
    async def decide(self, request: DecisionRequest) -> ActionDecision:
        """Return a decision for the current request."""


def metadata_from_policy_config(policy_id: str, policy: "PolicyConfig") -> PolicyAdapterMetadata:
    return PolicyAdapterMetadata(
        policy_id=policy_id,
        provider=policy.provider,
        model=policy.model,
        prompt_version=policy.prompt_version,
        options=policy.options,
    )


def build_policy_registry(runtime_config: "DaemonRuntimeConfig") -> Mapping[str, PolicyAdapter]:
    from yomi_daemon.adapters.anthropic import build_anthropic_adapter
    from yomi_daemon.adapters.baseline import build_baseline_adapter
    from yomi_daemon.adapters.openai import build_openai_adapter
    from yomi_daemon.adapters.openrouter import build_openrouter_adapter
    from yomi_daemon.adapters.rl import build_rl_adapter

    registry: dict[str, PolicyAdapter] = {}
    for policy_id, policy in runtime_config.policies.items():
        if policy.provider == "baseline":
            registry[policy_id] = build_baseline_adapter(
                policy_id,
                policy,
                default_trace_seed=runtime_config.trace_seed,
            )
            continue
        if policy.provider == "anthropic":
            registry[policy_id] = build_anthropic_adapter(
                policy_id,
                policy,
                decision_timeout_ms=runtime_config.decision_timeout_ms,
                fallback_mode=runtime_config.fallback_mode,
                default_trace_seed=runtime_config.trace_seed,
            )
            continue
        if policy.provider == "openai":
            registry[policy_id] = build_openai_adapter(
                policy_id,
                policy,
                decision_timeout_ms=runtime_config.decision_timeout_ms,
                fallback_mode=runtime_config.fallback_mode,
                default_trace_seed=runtime_config.trace_seed,
            )
            continue
        if policy.provider == "openrouter":
            registry[policy_id] = build_openrouter_adapter(
                policy_id,
                policy,
                decision_timeout_ms=runtime_config.decision_timeout_ms,
                fallback_mode=runtime_config.fallback_mode,
                default_trace_seed=runtime_config.trace_seed,
            )
            continue
        if policy.provider == "rl":
            registry[policy_id] = build_rl_adapter(
                policy_id,
                policy,
                decision_timeout_ms=runtime_config.decision_timeout_ms,
                fallback_mode=runtime_config.fallback_mode,
                default_trace_seed=runtime_config.trace_seed,
            )
            continue
        raise AdapterConstructionError(
            f"policy {policy_id!r} uses unsupported provider {policy.provider!r}; "
            "provider-backed adapters land in later work units"
        )
    return MappingProxyType(registry)


def build_player_policy_adapters(
    runtime_config: "DaemonRuntimeConfig",
    *,
    registry: Mapping[str, PolicyAdapter] | None = None,
) -> Mapping[PlayerSlot, PolicyAdapter]:
    resolved_registry = registry or build_policy_registry(runtime_config)
    slot_mapping = runtime_config.policy_mapping
    assignments = {
        PlayerSlot.P1: resolved_registry[slot_mapping.p1],
        PlayerSlot.P2: resolved_registry[slot_mapping.p2],
    }
    return MappingProxyType(assignments)
