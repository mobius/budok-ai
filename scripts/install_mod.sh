#!/usr/bin/env bash
# Install the packaged YomiLLMBridge mod into a YOMI Hustle game directory.
#
# Usage:
#   scripts/install_mod.sh --game-dir /path/to/yomi-hustle
#
# The script copies dist/YomiLLMBridge.zip into <game-dir>/mods/ and
# extracts it so the game's mod loader can discover it at startup.
#
# Prerequisites:
#   - Run scripts/package_mod.sh first to create the mod zip.
#   - The game directory must contain project.godot or project.binary.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOD_ZIP="$REPO_ROOT/dist/YomiLLMBridge.zip"

# --- Parse arguments ---

GAME_DIR=""

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
            printf 'Usage: %s --game-dir /path/to/yomi-hustle\n' "$0"
            exit 0
            ;;
        *)
            printf 'ERROR: Unknown argument: %s\n' "$1" >&2
            printf 'Usage: %s --game-dir /path/to/yomi-hustle\n' "$0" >&2
            exit 1
            ;;
    esac
done

if [ -z "$GAME_DIR" ]; then
    printf 'ERROR: --game-dir is required.\n' >&2
    printf 'Usage: %s --game-dir /path/to/yomi-hustle\n' "$0" >&2
    exit 1
fi

# --- Validate game directory ---

if [ ! -d "$GAME_DIR" ]; then
    printf 'ERROR: Game directory does not exist: %s\n' "$GAME_DIR" >&2
    exit 1
fi

# Check for a Godot project marker. Steam/exported builds ship the project data
# inside the .pck and may not have project.godot/project.binary.
GAME_BINARY="$GAME_DIR/YourOnlyMoveIsHUSTLE.x86_64"
GAME_PCK="$GAME_DIR/YourOnlyMoveIsHUSTLE.pck"
if [ ! -f "$GAME_DIR/project.godot" ] && [ ! -f "$GAME_DIR/project.binary" ] && \
   [ ! -f "$GAME_BINARY" ]; then
    printf 'ERROR: Directory does not look like a YOMI Hustle installation.\n' >&2
    printf '  Expected project.godot, project.binary, or %s in: %s\n' "$(basename "$GAME_BINARY")" "$GAME_DIR" >&2
    printf '  Point --game-dir to the directory containing the game executable.\n' >&2
    exit 1
fi

# --- Validate mod zip ---

if [ ! -f "$MOD_ZIP" ]; then
    printf 'ERROR: Mod zip not found: %s\n' "$MOD_ZIP" >&2
    printf '  Run scripts/package_mod.sh first to create the mod package.\n' >&2
    exit 1
fi

# --- Install ---

MODS_DIR="$GAME_DIR/mods"
mkdir -p "$MODS_DIR"

# Copy zip for the mod loader
cp "$MOD_ZIP" "$MODS_DIR/YomiLLMBridge.zip"

# Also extract so the mod is available as a directory (some mod loaders prefer this)
rm -rf "$MODS_DIR/YomiLLMBridge"
unzip -qo "$MOD_ZIP" -d "$MODS_DIR"

printf 'Installed mod to: %s/YomiLLMBridge\n' "$MODS_DIR"
printf 'Installed zip to: %s/YomiLLMBridge.zip\n' "$MODS_DIR"
