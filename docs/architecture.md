# Architecture

This document covers daemon and mod runtime architecture. The normative source remains [the unified spec](../specs/unified_spec.md).

## Ownership Boundaries

- `daemon/src/yomi_daemon/` owns orchestration, protocol handling, adapters, storage, and tournament operations.
- `mod/YomiLLMBridge/` owns live game integration and mod-side safety checks.
- `schemas/` owns the versioned transport contract.

## Daemon Module Map

| Module | Responsibility |
|---|---|
| `cli.py` | CLI entry point and argument parsing |
| `config.py` | Runtime config loading, validation, and precedence resolution |
| `server.py` | WebSocket server, handshake, and message routing |
| `orchestrator.py` | Match orchestration and turn lifecycle coordination |
| `match.py` | Match metadata and state tracking |
| `protocol.py` | Typed protocol models, enums, envelope contract, canonical hashing |
| `validation.py` | Schema validation and decision legality checks |
| `prompt.py` | Deterministic prompt rendering from templates |
| `response_parser.py` | Multi-format response parsing (structured, JSON, text) with correction retry |
| `fallback.py` | Fallback action selection for timeouts and invalid responses |
| `manifest.py` | Match manifest construction for reproducibility |
| `ids.py` | Deterministic ID and seed generation |
| `redact.py` | API key pattern redaction for logs and artifacts |
| `tooling.py` | `uv` environment verification entry point |
| `adapters/` | Provider and baseline policy adapters |
| `storage/` | Artifact writing and replay index construction |
| `logging/` | Structured logging schemas |
| `replay_capture.py` | FFmpeg-based replay video recording via OrbStack VM |
| `tournament/` | Round-robin scheduling, match execution, Elo ratings, reporting |

## Adapter Registry

`adapters/base.py` defines the `PolicyAdapter` protocol and `build_policy_registry()` which maps provider names to adapter constructors. Implemented adapters:

- `baseline.py` — four deterministic baselines (`random`, `block_always`, `greedy_damage`, `scripted_safe`)
- `anthropic.py` — Anthropic Claude models
- `openai.py` — OpenAI models
- `openrouter.py` — OpenRouter-proxied models, and OpenAI-compatible endpoints such as DeepSeek via `options.base_url`

The `openrouter` adapter supports these policy options:

| Option | Description |
|---|---|
| `base_url` | Override the API endpoint (e.g. `https://api.deepseek.com` for DeepSeek) |
| `response_format` | `json_schema` (default) or `json_object` for providers without schema support |
| `http_referer` | OpenRouter HTTP-Referer header |
| `title` | OpenRouter X-Title header |
| `categories` | OpenRouter X-OpenRouter-Categories header |
| `reasoning_effort` | Reasoning effort for supported models |

Placeholder stubs exist for `google`, `local`, and `ollama` but are not registered.

## Storage And Artifacts

`storage/writer.py` writes per-match artifacts to `runs/<timestamp>_<match_id>/`. `storage/replay_index.py` builds per-turn pointers into decisions and prompts for replay analysis.

## Tournament Subsystem

`tournament/` provides:

- `scheduler.py` — round-robin pairing generation with mirror-match and side-swap support
- `runner.py` — concurrent match execution against the daemon server
- `ratings.py` — Elo rating table computation from match results
- `reporter.py` — human-readable tournament report generation
- `cli.py` — `yomi-tournament` CLI with `schedule`, `report`, and `recompute` subcommands

## Mod Bridge Files

| File | Responsibility |
|---|---|
| `ModMain.gd` | Mod entry point and lifecycle |
| `TurnHook.gd` | Decision interception, turn lifecycle coordination |
| `ObservationBuilder.gd` | Deterministic game state serialization |
| `LegalActionBuilder.gd` | Legal action enumeration from UI state |
| `DecisionValidator.gd` | Action legality and schema validation |
| `ActionApplier.gd` | Native action application via `on_action_selected()` |
| `AutoMatchStarter.gd` | Programmatic match start from daemon config (bypasses menu UI) |
| `FallbackHandler.gd` | Timeout and error fallback action selection |
| `BridgeClient.gd` | WebSocket transport to daemon |
| `ProtocolCodec.gd` | Envelope encoding/decoding and canonical hashing |
| `IdentifierFactory.gd` | Match/turn ID minting |
| `RuntimeCompatibility.gd` | Game version and signal compatibility checks |
| `Telemetry.gd` | Lifecycle event emission |

Config files: `config/ModOptions.gd`, `config/default_config.json`.

## Daemon Runtime Config

`schemas/daemon-config.v1.json` validates daemon JSON config files before any match starts. `daemon/src/yomi_daemon/config.py` normalizes that file into a typed `DaemonRuntimeConfig`.

`DaemonRuntimeConfig` fields:

- `version` — config schema version (`"v1"`)
- `transport` — host, port, optional `auth_secret` (resolved from env var)
- `decision_timeout_ms` — decision timeout in milliseconds (default 10000)
- `fallback_mode` — `safe_continue`, `heuristic_guard`, or `last_valid_replayable`
- `logging` — events, prompts, raw_provider_payloads flags
- `policy_mapping` — p1/p2 policy ID assignment
- `policies` — registry of named policies with provider, model, prompt_version, credential, temperature, max_tokens, options
- `character_selection` — mode (mirror/assigned/random_from_pool/llm_choice), assignments, pool
- `tournament` — format, mirror_matches_first, side_swap, games_per_pair, fixed_stage
- `trace_seed` — reproducibility seed
- `stage_id` — optional stage override
- `match_options` — optional match configuration (e.g. `starting_hp`)
- `replay_capture` — replay video recording settings (enabled, vm_machine, display, resolution, framerate, video_codec, preset); default enabled

`DaemonRuntimeConfig.to_config_payload()` emits the narrower `ConfigPayload` used for handshake pinning and manifests. The `effective_stage_id` property resolves from `stage_id` or `tournament.fixed_stage`.

The split is deliberate: runtime config includes daemon concerns (transport, credentials, tournament defaults), while wire config stays safe to share with the mod.

## Config Precedence

Daemon startup resolves config in this order:

1. Built-in defaults in `yomi_daemon.config` for transport, logging, decision timeout, character selection, tournament defaults, and `trace_seed`.
2. A selected JSON config file, defaulting to `daemon/config/default_config.json`.
3. CLI overrides from `yomi-daemon --host/--port/--p1-policy/--p2-policy/--trace-seed/--log-level/--record-replay/--no-record-replay/--replay-vm/--replay-display`.
4. Environment lookup for provider credentials referenced by `credential_env_var`.

Environment variables only supply secret values. They never change structural settings such as policy IDs or transport bindings.

## Manifest

`daemon/src/yomi_daemon/manifest.py` builds a serializable `MatchManifest` before the first decision turn. The manifest pins:

- `match_id` and `created_at` timestamp
- `daemon_version` (from package `__version__`)
- `protocol_version` and `schema_version`
- `trace_seed`
- `game_version` and `mod_version` (from handshake metadata, nullable)
- `effective_stage_id`
- `prompt_version` (pinned when all policies share the same version, else null)
- `policy_mapping` and per-policy `ManifestPolicyEntry` (provider, model, prompt_version, credential_env_var, credential_configured)
- `transport` (host + port, no auth secret)
- `tournament` defaults
- full `config_snapshot` (`ConfigPayload`)

This ensures failed or incomplete matches still retain pinned config, version, and seed metadata.
