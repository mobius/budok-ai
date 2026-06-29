#!/usr/bin/env bash
# End-to-end match runner for budok-ai on Linux (native, no VM).
#
# Automates: kill stale game, ensure Xvfb, package/install mod,
# start daemon, launch game, wait for completion, report results.
#
# Usage:
#   scripts/run_match_linux.sh [CONFIG_FILE] [OPTIONS]
#
# Arguments:
#   CONFIG_FILE              Path to a match.conf file (default: match-linux.conf)
#
# Options (override config file values):
#   --daemon-config PATH     Daemon runtime config JSON
#   --log-level LVL          Log verbosity
#   --no-replay              Disable replay recording
#   --skip-mod-install       Skip mod packaging and installation (use existing mod)
#   --dry-run                Print what would be done, don't execute
#   -h, --help               Show this help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ─── Defaults ────────────────────────────────────────────────────────────────

GAME_DIR="${GAME_DIR:-/home/${USER:-$(whoami)}/games/yomi}"
DISPLAY_NUM=":99"
RESOLUTION="1280x720"
DAEMON_CONFIG="daemon/config/llm_v_llm.json"
DAEMON_PORT=""
LOG_LEVEL="INFO"
RECORD_REPLAY=true
ENV_FILE=".env"
SKIP_MOD_INSTALL=false
DRY_RUN=false

# ─── Parse arguments ─────────────────────────────────────────────────────────

CONF_FILE=""
EXTRA_DAEMON_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --daemon-config)   DAEMON_CONFIG="$2"; shift 2 ;;
        --daemon-config=*) DAEMON_CONFIG="${1#*=}"; shift ;;
        --log-level)       LOG_LEVEL="$2"; shift 2 ;;
        --log-level=*)     LOG_LEVEL="${1#*=}"; shift ;;
        --no-replay)       RECORD_REPLAY=false; shift ;;
        --skip-mod-install) SKIP_MOD_INSTALL=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --match-history)   EXTRA_DAEMON_ARGS+=("--match-history" "$2"); shift 2 ;;
        --match-history=*) EXTRA_DAEMON_ARGS+=("--match-history" "${1#*=}"); shift ;;
        -h|--help)         head -25 "$0" | tail -23; exit 0 ;;
        -*)                EXTRA_DAEMON_ARGS+=("$1"); shift ;;
        *)
            if [ -z "$CONF_FILE" ]; then
                CONF_FILE="$1"
            else
                EXTRA_DAEMON_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# ─── Load config file ────────────────────────────────────────────────────────

if [ -z "$CONF_FILE" ]; then
    if [ -f "$REPO_ROOT/match-linux.conf" ]; then
        CONF_FILE="$REPO_ROOT/match-linux.conf"
    elif [ -f "$REPO_ROOT/match.conf" ]; then
        CONF_FILE="$REPO_ROOT/match.conf"
    fi
fi

if [ -n "$CONF_FILE" ]; then
    if [ ! -f "$CONF_FILE" ]; then
        printf 'ERROR: Config file not found: %s\n' "$CONF_FILE" >&2
        exit 1
    fi
    printf 'Loading config: %s\n' "$CONF_FILE"
    while IFS='=' read -r key value; do
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        key="$(echo "$key" | xargs)"
        value="$(echo "$value" | xargs)"
        [ -z "$key" ] && continue
        eval "$key=\"$value\""
    done < "$CONF_FILE"
fi

# ─── Resolve paths ───────────────────────────────────────────────────────────

[[ "$DAEMON_CONFIG" != /* ]] && DAEMON_CONFIG="$REPO_ROOT/$DAEMON_CONFIG"
[[ "$ENV_FILE" != /* ]] && ENV_FILE="$REPO_ROOT/$ENV_FILE"

# ─── Utility functions ───────────────────────────────────────────────────────

log() { printf '[run_match_linux] %s\n' "$*"; }
err() { printf '[run_match_linux] ERROR: %s\n' "$*" >&2; }
warn() { printf '[run_match_linux] WARNING: %s\n' "$*" >&2; }

# ─── Dry run summary ─────────────────────────────────────────────────────────

if [ "$DRY_RUN" = true ]; then
    GAME_BINARY="$GAME_DIR/YourOnlyMoveIsHUSTLE.x86_64"
    log "DRY RUN — would execute the following steps:"
    log "  1. Kill stale game processes"
    log "  2. Ensure Xvfb on display $DISPLAY_NUM"
    log "  3. Package and install mod into $GAME_DIR/mods/"
    log "  4. Start daemon with config $DAEMON_CONFIG"
    log "  5. Launch game from $GAME_BINARY"
    log "  6. Wait for match completion"
    exit 0
fi

# ─── Pre-flight checks ───────────────────────────────────────────────────────

log "Pre-flight checks..."

for cmd in uv ffmpeg Xvfb xdpyinfo; do
    if ! command -v "$cmd" &>/dev/null; then
        err "$cmd is not installed"
        exit 1
    fi
done

if [ ! -f "$DAEMON_CONFIG" ]; then
    err "Daemon config not found: $DAEMON_CONFIG"
    exit 1
fi

if [ ! -d "$GAME_DIR" ]; then
    err "Game directory not found: $GAME_DIR"
    exit 1
fi

GAME_BINARY="$GAME_DIR/YourOnlyMoveIsHUSTLE.x86_64"
if [ ! -f "$GAME_BINARY" ]; then
    err "Game binary not found at $GAME_BINARY"
    exit 1
fi

if [ -f "$ENV_FILE" ]; then
    log "Loading env from $ENV_FILE"
    set -a
    while IFS= read -r line; do
        [[ "$line" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$line" ]] && continue
        eval "export $line" 2>/dev/null || true
    done < "$ENV_FILE"
    set +a
else
    warn "No .env file found at $ENV_FILE"
fi

# ─── Step 1: Kill stale game processes ────────────────────────────────────────

log "Killing stale game processes..."
pkill -f "YourOnlyMoveIsHUSTLE.x86_64" 2>/dev/null || true

# ─── Step 2: Ensure Xvfb is running ──────────────────────────────────────────

log "Ensuring Xvfb on display $DISPLAY_NUM..."
if ! DISPLAY="$DISPLAY_NUM" xdpyinfo &>/dev/null; then
    log "Starting Xvfb..."
    Xvfb "$DISPLAY_NUM" -screen 0 "${RESOLUTION}x24" -nocursor &>/dev/null &
    sleep 1
    if ! DISPLAY="$DISPLAY_NUM" xdpyinfo &>/dev/null; then
        rm -f /tmp/.X*-lock /tmp/.X11-unix/X* 2>/dev/null || true
        Xvfb "$DISPLAY_NUM" -screen 0 "${RESOLUTION}x24" -nocursor &>/dev/null &
        sleep 1
    fi
fi

if ! DISPLAY="$DISPLAY_NUM" xdpyinfo &>/dev/null; then
    err "Xvfb failed to start on display $DISPLAY_NUM"
    exit 1
fi

# ─── Step 3: Package and install mod ─────────────────────────────────────────

if [ "$SKIP_MOD_INSTALL" = false ]; then
    log "Packaging mod..."
    scripts/package_mod.sh

    log "Installing mod into $GAME_DIR/mods/..."
    scripts/install_mod.sh --game-dir "$GAME_DIR"
else
    log "Skipping mod install (--skip-mod-install)"
fi

# ─── Step 4: Start daemon ────────────────────────────────────────────────────

log "Starting daemon..."

DAEMON_ARGS=("--config" "$DAEMON_CONFIG" "--host" "0.0.0.0" "--log-level" "$LOG_LEVEL")

if [ -n "$DAEMON_PORT" ]; then
    DAEMON_ARGS+=("--port" "$DAEMON_PORT")
fi

if [ "$RECORD_REPLAY" = "false" ]; then
    DAEMON_ARGS+=("--no-record-replay")
else
    DAEMON_ARGS+=("--record-replay")
fi

# Local mode: run ffmpeg on the host instead of inside an OrbStack VM.
DAEMON_ARGS+=("--replay-vm" "local" "--replay-display" "$DISPLAY_NUM")
DAEMON_ARGS+=("${EXTRA_DAEMON_ARGS[@]+"${EXTRA_DAEMON_ARGS[@]}"}")

DAEMON_PID=""

cleanup() {
    if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        log "Stopping daemon (pid $DAEMON_PID)..."
        kill "$DAEMON_PID" 2>/dev/null || true
        sleep 1
        kill -9 "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    # Kill game process
    pkill -f "YourOnlyMoveIsHUSTLE.x86_64" 2>/dev/null || true
}
trap cleanup EXIT

MATCH_START_EPOCH=$(date +%s)

uv run --project daemon yomi-daemon "${DAEMON_ARGS[@]}" &
DAEMON_PID=$!

# Wait for daemon to start listening
LISTEN_PORT="${DAEMON_PORT:-8765}"
log "Waiting for daemon to listen on port $LISTEN_PORT..."
for i in $(seq 1 15); do
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        err "Daemon exited unexpectedly"
        exit 1
    fi
    if python3 -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(0.5)
s.connect(('127.0.0.1', int(sys.argv[1])))
s.close()
" "$LISTEN_PORT" 2>/dev/null; then
        break
    fi
    sleep 1
done

if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    err "Daemon exited before accepting connections"
    exit 1
fi

log "Daemon listening on ws://0.0.0.0:$LISTEN_PORT"

# ─── Step 5: Launch game ─────────────────────────────────────────────────────

log "Launching game..."

# Ensure the game binary is executable (DepotDownloader may not preserve +x).
chmod +x "$GAME_BINARY" 2>/dev/null || true

export LIBGL_ALWAYS_SOFTWARE=1
export LD_LIBRARY_PATH="$GAME_DIR:${LD_LIBRARY_PATH:-}"

GAME_LOG="/tmp/yomi_game_${GAME_PID:-$$}.log"
DISPLAY="$DISPLAY_NUM" "$GAME_BINARY" >"$GAME_LOG" 2>&1 &
GAME_PID=$!

log "Game launched (pid $GAME_PID). Waiting for match to complete..."

# ─── Step 6: Wait for match completion ────────────────────────────────────────

RESULT_FILE=""
LATEST_RUN=""
while true; do
    if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
        break
    fi

    for d in $(ls -td runs/*/ 2>/dev/null); do
        dir_epoch=$(stat -c %Y "$d" 2>/dev/null || echo 0)
        if [ "$dir_epoch" -ge "$MATCH_START_EPOCH" ]; then
            LATEST_RUN="$d"
            break
        fi
    done

    if [ -n "$LATEST_RUN" ] && [ -f "${LATEST_RUN}result.json" ] && \
       python3 -c "import json,sys; r=json.load(open(sys.argv[1])); sys.exit(0 if r.get('status') in ('completed','failed') else 1)" "${LATEST_RUN}result.json" 2>/dev/null; then
        RESULT_FILE="${LATEST_RUN}result.json"
        if [ "$RECORD_REPLAY" = "true" ]; then
            log "Match result found, waiting for replay recording..."
            for i in $(seq 1 180); do
                if [ -f "${LATEST_RUN}replay.mp4" ]; then
                    log "Replay video ready"
                    break
                fi
                if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
        fi
        sleep 3
        break
    fi
    sleep 2
done

if kill -0 "$DAEMON_PID" 2>/dev/null; then
    log "Stopping daemon..."
    kill "$DAEMON_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$DAEMON_PID" 2>/dev/null || true
fi
wait "$DAEMON_PID" 2>/dev/null || true
DAEMON_PID=""

# Report results
if [ -n "$RESULT_FILE" ]; then
    MATCH_STATUS=$(python3 -c "import json; print(json.load(open('$RESULT_FILE')).get('status','?'))" 2>/dev/null || echo "?")
    if [ "$MATCH_STATUS" = "failed" ]; then
        log "Match failed (game disconnected or crashed)"
    else
        log "Match completed successfully!"
    fi
    log "Artifacts: $LATEST_RUN"
    printf '\n'
    printf '╔══════════════════════════════════╗\n'
    printf '║         MATCH RESULT             ║\n'
    printf '╠══════════════════════════════════╣\n'
    python3 -c "
import json, sys
r = json.load(open(sys.argv[1]))
print(f'║  Status:  {r.get(\"status\", \"unknown\"):<21s} ║')
print(f'║  Winner:  {r.get(\"winner\", \"unknown\"):<21s} ║')
print(f'║  Reason:  {r.get(\"end_reason\", \"unknown\"):<21s} ║')
print(f'║  Turns:   {str(r.get(\"total_turns\", \"unknown\")):<21s} ║')
" "$RESULT_FILE" 2>/dev/null || true
    printf '╚══════════════════════════════════╝\n'
    if [ -f "${LATEST_RUN}replay.mp4" ]; then
        log "Replay video: ${LATEST_RUN}replay.mp4"
    fi
else
    err "Match did not complete — no result.json found"
    exit 1
fi
