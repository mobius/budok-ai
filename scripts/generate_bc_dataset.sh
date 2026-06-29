#!/usr/bin/env bash
# Generate trajectory dataset for behavior cloning by running baseline matches.
#
# Usage:
#   scripts/generate_bc_dataset.sh --game-dir /path/to/YomiHustle --count 10

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

GAME_DIR=""
COUNT=10
CONFIGS=(
    "daemon/config/bc_scripted_vs_random.json"
    "daemon/config/bc_greedy_vs_random.json"
)

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
        --count)
            COUNT="$2"
            shift 2
            ;;
        --count=*)
            COUNT="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --game-dir /path/to/YomiHustle [--count N]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$GAME_DIR" ]; then
    echo "ERROR: --game-dir is required"
    exit 1
fi

if ! command -v podman &>/dev/null; then
    echo "ERROR: podman is not installed"
    exit 1
fi

for config in "${CONFIGS[@]}"; do
    echo "=== Generating $COUNT matches with $config ==="
    for i in $(seq 1 "$COUNT"); do
        echo "Run $i/$COUNT ..."
        scripts/run_match_podman.sh --game-dir "$GAME_DIR" --daemon-config "$config" \
            --no-replay >/dev/null 2>&1 || true
    done
done

echo "Dataset generation complete. Artifacts in runs/"
