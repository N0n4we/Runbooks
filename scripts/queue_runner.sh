#!/usr/bin/env bash
# LoraCI task queue runner.
#
# Runs as a long-lived background process (started by run_training.sh under
# setsid+nohup). Each iteration:
#   1. git fetch + reset --hard origin/main  (picks up newly committed tasks)
#   2. ensure base models are present
#   3. find the lexicographically first tasks/*.toml whose
#      logs/done/<basename>__<sha8>.done marker does not exist
#   4. run python train.py on it; train.py emails its own success/failure
#   5. write the marker (regardless of train.py exit code) so failed tasks
#      don't block the queue. To re-run a failed task, edit the toml (hash
#      changes) or rename it.
#
# Exits 0 when the queue is empty. Sends a separate "queue runner crashed"
# email if the runner itself fails (git/network/disk) — per-task failures
# are reported by train.py.
#
# The script is wrapped in `{ ... }` so a `git reset --hard` mid-loop that
# rewrites this file cannot cause bash to read garbled bytes.
{
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

LOG_DIR="$REPO_DIR/logs"
DONE_DIR="$LOG_DIR/done"
TASKS_DIR="$REPO_DIR/tasks"
MODEL_DIR="$REPO_DIR/models/base"
PID_FILE="$LOG_DIR/queue_runner.pid"

mkdir -p "$LOG_DIR" "$DONE_DIR" "$TASKS_DIR" "$MODEL_DIR"

current_stage="init"
last_task=""

# cleanup is invoked via `trap cleanup EXIT` below.
# shellcheck disable=SC2329
cleanup() {
    local code=$?
    rm -f "$PID_FILE"

    if [ "$code" -ne 0 ]; then
        local body
        body="$(printf 'Host: %s\nStage: %s\nLast task: %s\nExit code: %s\n\nTail of queue_runner.log:\n%s\n' \
            "$(hostname)" \
            "$current_stage" \
            "${last_task:-<none>}" \
            "$code" \
            "$(tail -n 100 "$LOG_DIR/queue_runner.log" 2>/dev/null || echo '<unreadable>')")"
        python "$REPO_DIR/scripts/notify_email.py" \
            "[LoraCI] queue runner crashed" \
            "$body" || true
    fi
    exit "$code"
}
trap cleanup EXIT

# First 8 hex chars of sha256(file). Used to fingerprint task content so an
# edited toml gets re-run automatically.
sha8() {
    sha256sum "$1" | cut -c1-8
}

while true; do
    current_stage="git-sync"
    echo "=== [$(date -Iseconds)] Syncing repository ==="
    git fetch origin main
    git reset --hard origin/main

    current_stage="download-models"
    echo "=== [$(date -Iseconds)] Ensuring base models present ==="
    bash "$REPO_DIR/scripts/download_models.sh" "$MODEL_DIR"

    current_stage="scan-queue"
    next_path=""
    next_hash=""
    next_basename=""
    while IFS= read -r f; do
        [ -e "$f" ] || continue
        h=$(sha8 "$f")
        b=$(basename "$f" .toml)
        # A task is "already handled" if any marker exists for this exact
        # content hash — covers .done (ran) and .missing_dataset (skipped).
        if compgen -G "$DONE_DIR/${b}__${h}."* > /dev/null; then
            continue
        fi
        next_path="$f"
        next_hash="$h"
        next_basename="$b"
        break
    done < <(find "$TASKS_DIR" -maxdepth 1 -type f -name '*.toml' | sort)

    if [ -z "$next_path" ]; then
        echo "=== [$(date -Iseconds)] Queue empty, runner exiting cleanly ==="
        break
    fi

    current_stage="train:$next_basename"
    last_task="$next_basename"
    ts=$(date +%Y%m%d_%H%M%S)
    train_log="$LOG_DIR/train_${next_basename}_${ts}.log"

    # Convention: dataset for tasks/<name>.toml lives in tasks/<name>/.
    # That directory is gitignored — operators rsync the data onto the
    # training host themselves. Skip the task (with a clear email) if the
    # directory is missing or empty, instead of letting train.py die deep
    # inside the dataloader.
    dataset_dir="$TASKS_DIR/$next_basename"
    if [ ! -d "$dataset_dir" ] || [ -z "$(find "$dataset_dir" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        echo "=== Task $next_basename SKIPPED: dataset $dataset_dir missing or empty ==="
        body="$(printf 'Host: %s\nTask: %s\nExpected dataset directory: %s\n\nThe directory is missing or contains no files.\nrsync your training data into that path on the training host, then push a change to %s.toml (any edit changes its hash and re-queues it).\n' \
            "$(hostname)" \
            "$next_basename" \
            "$dataset_dir" \
            "$next_basename")"
        python "$REPO_DIR/scripts/notify_email.py" \
            "[LoraCI] $next_basename - dataset missing, task skipped" \
            "$body" || true
        # Mark with a distinct suffix so it's easy to tell missing-dataset
        # skips apart from completed/failed runs at a glance.
        touch "$DONE_DIR/${next_basename}__${next_hash}.missing_dataset"
        continue
    fi

    echo "=== [$(date -Iseconds)] Running task: $next_basename (sha=$next_hash) ==="

    # Disable -e for the single train.py call so a failing task doesn't
    # abort the runner. train.py emails its own failure notification.
    set +e
    python train.py --config_file "$next_path" > "$train_log" 2>&1
    rc=$?
    set -e

    # Mark processed regardless of outcome. To re-run, edit the toml
    # (hash changes -> new marker name) or rename the file.
    touch "$DONE_DIR/${next_basename}__${next_hash}.done"

    if [ "$rc" -eq 0 ]; then
        echo "=== Task $next_basename completed (log: $train_log) ==="
    else
        echo "=== Task $next_basename FAILED (rc=$rc, log: $train_log) ==="
    fi
done

# Clean exit: detach the trap and remove the PID file ourselves so
# cleanup() doesn't fire and email "crashed".
trap - EXIT
rm -f "$PID_FILE"
exit 0
}
