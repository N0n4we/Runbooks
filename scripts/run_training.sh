#!/usr/bin/env bash
# LoraCI training entry point.
#
# Ensures the queue runner is alive. The actual training loop lives in
# scripts/queue_runner.sh — this wrapper exists only to:
#   1. create the directories the runner expects
#   2. avoid spawning a second runner if one is already going
#   3. detach a fresh runner with setsid+nohup so it survives SSH close
#   4. email if any of the above fail
#
# Per-task failures are reported by train.py. Runner-level crashes are
# reported by queue_runner.sh's own EXIT trap.
#
# Required environment variables (passed in by the GitHub Actions SSH step):
#   SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, NOTIFY_EMAIL
{
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="$REPO_DIR/logs"
TASKS_DIR="$REPO_DIR/tasks"
MODEL_DIR="$REPO_DIR/models/base"
RUNNER_PID_FILE="$LOG_DIR/queue_runner.pid"
RUNNER_LOG="$LOG_DIR/queue_runner.log"

mkdir -p "$LOG_DIR" "$LOG_DIR/done" "$TASKS_DIR" "$MODEL_DIR"

PREFLIGHT_LOG="$LOG_DIR/preflight_$(date +%Y%m%d_%H%M%S).log"

run_preflight() {
    set -euo pipefail

    # If a runner is already running, leave it alone — it will git pull at
    # the top of its next loop iteration and pick up newly committed tasks.
    if [ -f "$RUNNER_PID_FILE" ]; then
        local prev
        prev=$(cat "$RUNNER_PID_FILE" 2>/dev/null || true)
        if [ -n "$prev" ] && kill -0 "$prev" 2>/dev/null; then
            echo "Queue runner already running (PID $prev). New tasks will be picked up automatically."
            return 0
        fi
        echo "[info] removing stale PID file (PID $prev no longer running)"
        rm -f "$RUNNER_PID_FILE"
    fi

    echo "=== [$(date -Iseconds)] Starting queue runner ==="
    # setsid + nohup + redirect of all standard fds: detach fully so the
    # runner survives SSH session termination and logind KillUserProcesses.
    setsid nohup bash "$REPO_DIR/scripts/queue_runner.sh" \
        < /dev/null > "$RUNNER_LOG" 2>&1 &
    local pid=$!
    disown "$pid" 2>/dev/null || true
    echo "$pid" > "$RUNNER_PID_FILE"

    echo "Queue runner launched (PID: $pid, log: $RUNNER_LOG)"
}

set +e
run_preflight 2>&1 | tee "$PREFLIGHT_LOG"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -ne 0 ]; then
    echo "Preflight failed with exit code $status, sending failure email..." >&2
    body="$(printf 'Host: %s\nRepo: %s\nExit code: %s\n\nPreflight log:\n%s\n' \
        "$(hostname)" \
        "$REPO_DIR" \
        "$status" \
        "$(cat "$PREFLIGHT_LOG" 2>/dev/null || echo '<preflight log unreadable>')")"
    python "$REPO_DIR/scripts/notify_email.py" \
        "[LoraCI] CI failed before queue runner started" \
        "$body" || true
    exit "$status"
fi

exit 0
}
