"""Opt-in smoke tests for real provider backends.

These tests are SKIPPED by default. To run them, set the environment variable
YOMI_SMOKE_PROVIDER to the provider name (e.g. "anthropic", "openai", "openrouter")
and ensure the corresponding API key env var is set:

    YOMI_SMOKE_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... uv run pytest tests/daemon/test_real_provider_smoke.py -v
    YOMI_SMOKE_PROVIDER=openai OPENAI_API_KEY=sk-... uv run pytest tests/daemon/test_real_provider_smoke.py -v
    YOMI_SMOKE_PROVIDER=openrouter OPENROUTER_API_KEY=sk-... uv run pytest tests/daemon/test_real_provider_smoke.py -v
    YOMI_SMOKE_PROVIDER=deepseek DEEPSEEK_API_KEY=sk-... uv run pytest tests/daemon/test_real_provider_smoke.py -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

from tests.daemon._decision_fixtures import build_action, build_request
from yomi_daemon.config import PolicyConfig, ProviderCredential
from yomi_daemon.protocol import FallbackMode

_SMOKE_PROVIDER = os.environ.get("YOMI_SMOKE_PROVIDER", "")

_PROVIDER_CONFIGS: dict[str, dict[str, object]] = {
    "anthropic": {
        "env_var": "ANTHROPIC_API_KEY",
        "model": "claude-haiku-4-5-20251001",
        "builder": "build_anthropic_adapter",
        "module": "yomi_daemon.adapters.anthropic",
    },
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "builder": "build_openai_adapter",
        "module": "yomi_daemon.adapters.openai",
    },
    "openrouter": {
        "env_var": "OPENROUTER_API_KEY",
        "model": "anthropic/claude-haiku-4-5-20251001",
        "builder": "build_openrouter_adapter",
        "module": "yomi_daemon.adapters.openrouter",
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "model": "deepseek-v4-flash",
        "builder": "build_openrouter_adapter",
        "module": "yomi_daemon.adapters.openrouter",
        "base_url": "https://api.deepseek.com",
    },
}

skip_unless_smoke = pytest.mark.skipif(
    not _SMOKE_PROVIDER,
    reason="YOMI_SMOKE_PROVIDER not set; opt-in smoke test",
)


def _resolve_provider_config() -> tuple[str, str, str, str]:
    """Return (provider, api_key, model, builder_func_name, module_name)."""
    if _SMOKE_PROVIDER not in _PROVIDER_CONFIGS:
        pytest.skip(f"Unknown smoke provider: {_SMOKE_PROVIDER!r}")
    config = _PROVIDER_CONFIGS[_SMOKE_PROVIDER]
    env_var = str(config["env_var"])
    api_key = os.environ.get(env_var, "")
    if not api_key:
        pytest.skip(f"{env_var} not set for smoke provider {_SMOKE_PROVIDER!r}")
    model = str(config["model"])
    return _SMOKE_PROVIDER, api_key, model, env_var


@skip_unless_smoke
def test_real_provider_returns_valid_decision() -> None:
    """Smoke test: a real provider returns a parseable action decision for a simple request."""
    provider, api_key, model, env_var = _resolve_provider_config()

    request = build_request(
        (
            build_action(
                "guard", description="Block incoming attacks safely.", di=True
            ),
            build_action("slash", description="Fast melee attack.", damage=60.0),
        ),
        deadline_ms=15000,
    )

    config = _PROVIDER_CONFIGS[provider]
    policy = PolicyConfig(
        provider=provider,
        model=model,
        prompt_version="minimal_v1",
        credential=ProviderCredential(env_var=env_var, value=api_key),
        temperature=0.0,
        max_tokens=128,
        options={"base_url": config["base_url"]} if "base_url" in config else {},
    )

    # Dynamic import to avoid import errors when provider SDK is not installed
    import importlib

    module = importlib.import_module(str(config["module"]))
    builder = getattr(module, str(config["builder"]))

    adapter = builder(
        f"smoke/{provider}",
        policy,
        decision_timeout_ms=15000,
        fallback_mode=FallbackMode.SAFE_CONTINUE,
    )

    result = asyncio.run(adapter.decide_with_trace(request))

    # The decision must reference a legal action
    assert result.decision.action in ("guard", "slash")
    assert result.decision.match_id == request.match_id
    assert result.decision.turn_id == request.turn_id

    # Provider trace must be present with at least one attempt
    assert result.prompt_trace is not None
    assert result.prompt_trace.prompt_text
    assert result.prompt_trace.prompt_version == "minimal_v1"
    assert result.prompt_trace.provider_request is not None
    assert result.prompt_trace.provider_response is not None


@skip_unless_smoke
def test_real_provider_captures_token_metadata() -> None:
    """Smoke test: token usage metadata is populated when the provider exposes it."""
    provider, api_key, model, env_var = _resolve_provider_config()

    request = build_request(
        (build_action("guard", description="Block."),),
        deadline_ms=15000,
    )

    config = _PROVIDER_CONFIGS[provider]
    policy = PolicyConfig(
        provider=provider,
        model=model,
        prompt_version="minimal_v1",
        credential=ProviderCredential(env_var=env_var, value=api_key),
        temperature=0.0,
        max_tokens=64,
        options={"base_url": config["base_url"]} if "base_url" in config else {},
    )

    import importlib

    module = importlib.import_module(str(config["module"]))
    builder = getattr(module, str(config["builder"]))

    adapter = builder(
        f"smoke/{provider}",
        policy,
        decision_timeout_ms=15000,
        fallback_mode=FallbackMode.SAFE_CONTINUE,
    )

    result = asyncio.run(adapter.decide_with_trace(request))

    # Token metadata should be populated for successful calls
    if result.decision.fallback_reason is None:
        assert result.decision.tokens_in is not None and result.decision.tokens_in > 0
        assert result.decision.tokens_out is not None and result.decision.tokens_out > 0
