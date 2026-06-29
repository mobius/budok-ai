#!/usr/bin/env bash
# End-to-end match runner for budok-ai using podman.
#
# This avoids installing Xvfb/Mesa/FFmpeg system-wide on the host.  Everything
# runs inside a container with its own Xvfb display.  The repository and the
# game directory are mounted into the container, so match artifacts are written
# back to the host's runs/ directory.
#
# Usage:
#   scripts/run_match_podman.sh --game-dir /path/to/yomi [run_match_linux args]
#
# Examples:
#   scripts/run_match_podman.sh --game-dir /home/joey/games/yomi
#   scripts/run_match_podman.sh --game-dir /home/joey/games/yomi --no-replay

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMAGE_NAME="budok-ai"
GAME_DIR=""
PASS_THROUGH_ARGS=()

log() { printf '[run_match_podman] %s\n' "$*"; }
err() { printf '[run_match_podman] ERROR: %s\n' "$*" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --game-dir)
            GAME_DIR="$2"
            shift 2
            ;;
        --game-dir=*)
            GAME_DIR="${1#*=}"
            shift
            ;;
        -h|--help)
            head -20 "$0" | tail -18
            exit 0
            ;;
        *)
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$GAME_DIR" ]; then
    err "--game-dir is required (path to YOMI Hustle installation on host)"
    exit 1
fi

if [ ! -d "$GAME_DIR" ]; then
    err "Game directory does not exist: $GAME_DIR"
    exit 1
fi

if ! command -v podman &>/dev/null; then
    err "podman is not installed"
    exit 1
fi

# Build the image if it doesn't exist.
if ! podman image exists "$IMAGE_NAME" >/dev/null 2>&1; then
    log "Building podman image $IMAGE_NAME..."
    podman build -t "$IMAGE_NAME" -f "$REPO_ROOT/Containerfile" "$REPO_ROOT"
fi

# Forward common API key env vars if present.
ENV_ARGS=()
for var in ANTHROPIC_API_KEY OPENAI_API_KEY OPENROUTER_API_KEY DEEPSEEK_API_KEY YOMI_AUTH_SECRET YOMI_SMOKE_PROVIDER; do
    if [ -n "${!var:-}" ]; then
        ENV_ARGS+=("--env" "$var=${!var}")
    fi
done

log "Starting budok-ai container..."
log "  Repository mount: $REPO_ROOT -> /budok-ai"
log "  Game mount:       $GAME_DIR -> /games/yomi"

exec podman run --rm --network=host \
    "${ENV_ARGS[@]+"${ENV_ARGS[@]}"}" \
    --env "UV_PROJECT_ENVIRONMENT=/opt/venv" \
    -v "$REPO_ROOT:/budok-ai:Z" \
    -v "$GAME_DIR:/games/yomi:Z" \
    -w /budok-ai \
    "$IMAGE_NAME" \
    bash -c 'set -euo pipefail; uv sync --project daemon --active; GAME_DIR=/games/yomi exec scripts/run_match_linux.sh "$@"' \
    bash "${PASS_THROUGH_ARGS[@]+"${PASS_THROUGH_ARGS[@]}"}"
