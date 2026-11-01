#!/bin/bash
# Chain several replacement tasks back to back on the server.
# usage: bash run-chains.sh <name> <task1-params> [<task2-params> ...]
#   <name>       label used for the input/output file names (taskN-<name>-first.png, chain/taskN-<name>)
#   each params  "<task> <size> <max-seconds> <deadline-hours>"
# Example:
#   bash run-chains.sh charA "task1 480x320 115 2.8" "task2 512x320 90 2.0" "task3 320x448 140 3.2"
# Started and stopped by hand; no auto-restart on failure. Requires the prompt/ref files below.
set -u
cd /root/h3-replace
PY=/root/h3-venv/bin/python
NAME=$1; shift
COMMON=(--seconds 10 --restart-each --anchor --max-jump 45 --first-check 0
        --ref-default "/root/h3-replace/chain/input/${NAME}/soft.png"
        --ref-full "/root/h3-replace/chain/input/${NAME}/full.png"
        --prompt-file "/root/h3-replace/prompt-${NAME}.txt")

for spec in "$@"; do
    set -- $spec
    task=$1 size=$2 secs=$3 hours=$4
    $PY -u chain-h3.py --task "$task" --variant "$NAME" --size "$size" \
        --out "chain/${task}-${NAME}" --max-seconds "$secs" --deadline-hours "$hours" \
        "${COMMON[@]}"
done
echo ALL-DONE
