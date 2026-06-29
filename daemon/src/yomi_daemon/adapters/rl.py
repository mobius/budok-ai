"""Reinforcement learning policy adapter.

Loads a trained PyTorch policy and serves actions via fast forward inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from yomi_daemon.adapters.base import (
    AdapterConstructionError,
    BasePolicyAdapter,
    PolicyAdapterMetadata,
    PolicyDecisionResult,
    PromptTrace,
    metadata_from_policy_config,
)
from yomi_daemon.protocol import ActionDecision, DecisionRequest, FallbackMode, LegalAction
from yomi_daemon.rl import ActionEncoder, ObservationEncoder
from yomi_daemon.rl.policy import YomiPolicy

if TYPE_CHECKING:
    from yomi_daemon.config import PolicyConfig


class RLAdapter(BasePolicyAdapter):
    """Adapter that uses a trained neural policy for decisions."""

    def __init__(
        self,
        *,
        metadata: PolicyAdapterMetadata,
        model_path: Path,
        observation_encoder: ObservationEncoder,
        action_encoder: ActionEncoder,
        deterministic: bool = True,
        default_trace_seed: int = 0,
    ) -> None:
        super().__init__(metadata=metadata, default_trace_seed=default_trace_seed)
        self._model = YomiPolicy.load(model_path)
        self._model.eval()
        self._observation_encoder = observation_encoder
        self._action_encoder = action_encoder
        self._deterministic = deterministic

    async def decide(self, request: DecisionRequest) -> ActionDecision:
        return (await self.decide_with_trace(request)).decision

    async def decide_with_trace(self, request: DecisionRequest) -> PolicyDecisionResult:
        obs_vector = self._observation_encoder.encode(request.observation)
        legal_indices = self._action_encoder.encode_legal_actions(request.legal_actions)
        legal_mask, _ = self._action_encoder.build_mask(legal_indices)

        action_index = self._model.select_action(
            obs_vector,
            legal_mask,
            deterministic=self._deterministic,
        )
        action_name, payload = self._action_encoder.decode_action(action_index)

        legal_action = self._find_legal_action(request, action_name)

        decision = self.build_decision(
            request,
            legal_action,
            data=payload or None,
            notes="RL policy decision",
        )

        return PolicyDecisionResult(
            decision=decision,
            prompt_trace=PromptTrace(
                prompt_text="",
                prompt_version=None,
                provider_request=None,
                provider_response=None,
            ),
        )

    def _find_legal_action(self, request: DecisionRequest, action_name: str) -> LegalAction:
        for legal_action in request.legal_actions:
            if legal_action.action == action_name:
                return legal_action
        # Fallback to the first legal action if the decoded name is not present
        # (should not happen with a consistent vocabulary).
        return request.legal_actions[0]


def build_rl_adapter(
    policy_id: str,
    policy: "PolicyConfig",
    *,
    decision_timeout_ms: int,
    fallback_mode: FallbackMode,
    default_trace_seed: int = 0,
) -> RLAdapter:
    """Build an RL adapter from policy config.

    Required policy options:
        - model_path: path to a saved YomiPolicy checkpoint

    Optional policy options:
        - encoder_dir: directory containing observation/action encoder configs
        - deterministic: whether to sample or take argmax (default: true)
    """
    del decision_timeout_ms, fallback_mode  # unused: RL adapter is fast and deterministic

    options = dict(policy.options)
    model_path_str = options.get("model_path")
    if not isinstance(model_path_str, str) or not model_path_str:
        raise AdapterConstructionError(
            f"rl policy {policy_id!r} must set options.model_path"
        )
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise AdapterConstructionError(
            f"rl policy {policy_id!r} model not found: {model_path}"
        )

    encoder_dir_str = options.get("encoder_dir")
    encoder_dir = Path(encoder_dir_str) if isinstance(encoder_dir_str, str) else model_path.parent

    observation_encoder = _load_observation_encoder(encoder_dir)
    action_encoder = _load_action_encoder(encoder_dir)

    metadata = metadata_from_policy_config(policy_id, policy)
    return RLAdapter(
        metadata=metadata,
        model_path=model_path,
        observation_encoder=observation_encoder,
        action_encoder=action_encoder,
        deterministic=bool(options.get("deterministic", True)),
        default_trace_seed=default_trace_seed,
    )


def _load_observation_encoder(encoder_dir: Path) -> ObservationEncoder:
    config_path = encoder_dir / "encoder_config.json"
    if config_path.exists():
        import json

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
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        action_to_index = config.get("action_encoder", {}).get("action_to_index")
        if isinstance(action_to_index, dict):
            encoder = ActionEncoder()
            # Re-register actions in index order
            for action_key, idx in sorted(action_to_index.items(), key=lambda kv: kv[1]):
                # action_key is canonical; decode enough to register
                if ":" in action_key:
                    action_name, _payload_json = action_key.split(":", 1)
                else:
                    action_name = action_key
                encoder.encode_action(action_name)
            return encoder
    return ActionEncoder()
