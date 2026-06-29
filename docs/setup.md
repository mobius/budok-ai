# Setup

This guide covers everything needed to develop and run experiments with the budok-ai.

## Prerequisites

| Dependency | Version | Required for |
|---|---|---|
| [`uv`](https://docs.astral.sh/uv/) | Latest | All Python commands |
| Python | 3.12+ | Daemon, tests, scripts |
| YOMI Hustle (Steam) | Build `16151810` | Live matches |
| Godot | 3.5.1 | Mod development, decompile inspection |
| GDRETools or gdsdecomp | Any recent | Game decompilation (optional) |

## 1. Clone and install dependencies

```bash
git clone <repo-url> budok-ai
cd budok-ai
uv sync --project daemon
```

Verify the install:

```bash
uv run --project daemon yomi-daemon --help
uv run --project daemon pytest tests/daemon -q
```

## 2. Provider credentials (optional)

Provider-backed policies (Anthropic, OpenAI, OpenRouter, DeepSeek) need API keys. Baseline policies need no credentials.

```bash
cp .env.example .env
# Edit .env and uncomment/fill the keys you need
```

Keys are resolved from environment variables at runtime. Set them however you prefer:

```bash
# Option A: source a .env file
export $(grep -v '^#' .env | xargs)

# Option B: inline
ANTHROPIC_API_KEY=sk-ant-... scripts/run_live_match.sh --p1-policy anthropic/claude
```

Available credential env vars:

| Provider | Env var |
|---|---|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

To test that credentials work:

```bash
YOMI_SMOKE_PROVIDER=anthropic uv run --project daemon pytest tests/daemon/test_real_provider_smoke.py -v
```

## 3. Shared-secret authentication (optional)

The daemon can require clients to authenticate during the handshake:

1. Set `transport.auth_secret_env_var` in the daemon config (e.g., `"YOMI_AUTH_SECRET"`).
2. Export the variable: `export YOMI_AUTH_SECRET=your-secret-here`.
3. Set `transport.auth_token` in the mod's bridge config to the same value.

Unauthenticated clients receive a WebSocket close with code 1008.

## 4. Game installation

YOMI Hustle is available on Steam (app `2212330`). The supported build is `16151810` with `Global.VERSION == "1.9.20-steam"` running on Godot `3.5.1`.

Default Steam install locations:

| Platform | Path |
|---|---|
| macOS | `~/Library/Application Support/Steam/steamapps/common/YomiHustle` |
| Linux | `~/.steam/steam/steamapps/common/YomiHustle` |
| Windows | `C:\Program Files (x86)\Steam\steamapps\common\YomiHustle` |

The mod bridge checks `Global.VERSION` and engine version at runtime. Other builds may work but are untested.

## 5. Mod packaging and installation

Package the mod from the repository:

```bash
scripts/package_mod.sh
```

This creates `dist/YomiLLMBridge.zip`.

Install it into the game:

```bash
scripts/install_mod.sh --game-dir /path/to/YomiHustle
```

This copies and extracts the mod into `<game-dir>/mods/YomiLLMBridge/`. The game's mod loader discovers it at startup.

## 6. Game decompilation (optional)

Decompilation is only needed if you're investigating game internals or validating hook points against a new build. It is not required for running experiments.

```bash
# Smoke mode: prepare the output layout only
scripts/decompile.sh

# Real decompile: requires GDRETools and a local copy of the game PCK
YOMI_PCK_PATH="/path/to/game.pck" \
GDRETOOLS_CMD="/path/to/gdre" \
scripts/decompile.sh --mode real
```

Output goes to `docs/decompile-output/`:

- `project/` — the decompiled Godot project
- `reports/` — symbol inventories and validation notes
- `manifest.json` — run metadata
- `reference-hooks.json` — hook inventory from reference mods

Open the decompiled project in Godot `3.5.1` only. Newer Godot versions may silently corrupt the project.

### SteamCMD notes (from WU-001)

- Anonymous SteamCMD access works for `app_info_print` but not depot download. Pulling the depot requires a Steam account that owns the game.
- On macOS, Homebrew-installed `steamcmd` may fail because Gatekeeper quarantines Steam runtime files. Clear `com.apple.quarantine` from `~/Library/Application Support/Steam/Steam.AppBundle` if this happens.

## 7. Daemon configuration

The default config lives at `daemon/config/default_config.json`. It runs two `baseline/random` policies with a 10-second decision timeout.

To customize, copy and edit:

```bash
cp daemon/config/default_config.json daemon/config/my_config.json
# Edit my_config.json
uv run --project daemon yomi-daemon --config daemon/config/my_config.json
```

See [`docs/architecture.md`](architecture.md#daemon-runtime-config) for all config fields and precedence rules.

### Adding a provider-backed policy

Add a policy entry to the `policies` object and reference it in `policy_mapping`:

```json
{
  "policy_mapping": {
    "p1": "anthropic/claude-sonnet",
    "p2": "baseline/greedy_damage"
  },
  "policies": {
    "anthropic/claude-sonnet": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "prompt_version": "strategic_v1",
      "credential_env_var": "ANTHROPIC_API_KEY"
    },
    "baseline/greedy_damage": {
      "provider": "baseline",
      "prompt_version": "none"
    }
  }
}
```

DeepSeek uses the `openrouter` adapter with a custom `base_url`:

```json
{
  "policies": {
    "deepseek/chat": {
      "provider": "openrouter",
      "model": "deepseek-v4-flash",
      "prompt_version": "strategic_v1",
      "credential_env_var": "DEEPSEEK_API_KEY",
      "temperature": 0.7,
      "max_tokens": 4096,
      "options": {
        "base_url": "https://api.deepseek.com",
        "response_format": "json_object"
      }
    }
  }
}
```

For LLM-backed policies, set `decision_timeout_ms` to 30000 to accommodate API latency. Decision requests for both players are processed concurrently. See `daemon/config/llm_first_test.json`, `daemon/config/llm_v_llm.json`, and `daemon/config/deepseek_vs_baseline.json` for complete examples.

## 8. Quality gates

Run before committing any changes:

```bash
uv run --project daemon ruff format
uv run --project daemon ruff check
uv run --project daemon ty check
uv run --project daemon pytest
```

## Directory layout reference

```
budok-ai/
  daemon/              Python daemon package
    config/            Runtime config files (default, llm_v_llm, llm_first_test)
    src/yomi_daemon/   Daemon source code
  mod/                 Godot mod sources
    YomiLLMBridge/     The bridge mod
      bridge/          GDScript bridge files
      config/          Mod config
  schemas/             Versioned JSON schemas (v1, v2)
  prompts/             Prompt templates and move catalog
  scripts/             Helper scripts
    run_match.sh       End-to-end match runner (macOS + OrbStack)
    run_live_match.sh  Manual daemon-only match runner
    package_mod.sh     Package mod into dist/YomiLLMBridge.zip
    install_mod.sh     Install mod into game directory
  match.conf.example   Config template for run_match.sh
  tests/               Test suites and fixtures
  runs/                Per-match artifact directories
  plans/               Work unit plans (v0.md, v1.md)
  specs/               Unified specification
  docs/                Documentation
```
