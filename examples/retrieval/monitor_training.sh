#!/usr/bin/env bash
# Periodic health monitor for the retrieval training queue.
#
# Appends a verdict for each pass so long unattended runs can be audited after
# the fact. Each pass checks that a training process is alive, that its
# optimizer step is advancing, that the loss is below the random baseline, and
# that the device is actually being used.
#
# Usage:
#   ./monitor_training.sh /path/to/monitor.log
#
# Env:
#   MONITOR_INTERVAL_SECONDS  seconds between passes (default 7200 = 2h)
#   RUN_ROOT                  run output root to inspect
#   QUEUE_LOG_GLOB            queue logs whose tails are shown when idle
set -u

OUT="${1:-/flagos-search-ckpts/torch-fl-cuda-3epoch-v2-monitor2h.log}"
INTERVAL="${MONITOR_INTERVAL_SECONDS:-7200}"
RUN_ROOT="${RUN_ROOT:-/flagos-search-ckpts/torch-fl-cuda-3epoch-v2}"
STATE="${STATE:-/tmp/retrieval-monitor.state}"
QUEUE_LOG_GLOB="${QUEUE_LOG_GLOB:-}"

# ln(2) is the expected loss for a balanced binary objective; the contrastive
# objectives sit well below it once training is working. A loss at or above it
# means the objective is not learning (this caught a reranker that was scoring
# through a randomly initialised head).
RANDOM_BASELINE="0.6931"

latest_log() {
  find "$RUN_ROOT" -path '*/attempt_0/*/stdout.log' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}

latest_step_and_loss() {
  grep -oE 'optimizer_step=[0-9]+ running_loss=[0-9.]+' "$1" 2>/dev/null | tail -1
}

mkdir -p "$(dirname "$OUT")"
while true; do
  {
    echo "==================== $(date -Is) ===================="

    PROCS=$(ps -eo pid,etime,%cpu,cmd \
      | grep -E 'flagscale.run|python -u flagscale/train/train_retrieval' \
      | grep -v grep || true)
    if [ -z "$PROCS" ]; then
      echo "VERDICT: NO_TRAINING_PROCESS (queue finished or died)"
      if [ -n "$QUEUE_LOG_GLOB" ]; then
        echo "-- queue log tail --"
        # shellcheck disable=SC2086
        tail -n 6 $QUEUE_LOG_GLOB 2>/dev/null | tail -n 12
      fi
    else
      echo "process: OK"
      echo "$PROCS" | sed 's/--config-path.*//' | cut -c1-110
    fi

    LOG=$(latest_log)
    if [ -n "$LOG" ]; then
      echo "log: $LOG"
      echo "model: $(echo "$LOG" | sed -E "s#^$RUN_ROOT/##; s#/.*##")"
      tail -n 3 "$LOG"
      CUR=$(latest_step_and_loss "$LOG")
      STEP=$(echo "$CUR" | grep -oE 'optimizer_step=[0-9]+' | cut -d= -f2 || true)
      LOSS=$(echo "$CUR" | grep -oE 'running_loss=[0-9.]+' | cut -d= -f2 || true)
      if [ -n "${STEP:-}" ]; then
        PREV=$(cat "$STATE" 2>/dev/null || true)
        PREV_STEP=$(echo "$PREV" | cut -d, -f2)
        PREV_TS=$(echo "$PREV" | cut -d, -f1)
        echo "step=$STEP loss=$LOSS"
        if [ -n "${PREV_STEP:-}" ] && [ "$STEP" = "$PREV_STEP" ]; then
          echo "VERDICT: STALLED (step unchanged since $PREV_TS)"
        elif [ -n "${PREV_STEP:-}" ]; then
          echo "VERDICT: PROGRESS (+$((STEP - PREV_STEP)) steps since last check)"
        else
          echo "VERDICT: PROGRESS (first check)"
        fi
        if awk -v l="$LOSS" -v b="$RANDOM_BASELINE" 'BEGIN{exit !(l < b)}'; then
          echo "loss: OK (below random baseline $RANDOM_BASELINE)"
        else
          echo "loss: SUSPECT (>= random baseline $RANDOM_BASELINE; objective may not be learning)"
        fi
        echo "$(date -Is),$STEP" > "$STATE"
      else
        echo "VERDICT: NO_STEP_LOGGED_YET"
      fi
    else
      echo "VERDICT: NO_LOG_FOUND"
    fi

    echo "-- gpu --"
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
      || echo "nvidia-smi unavailable"
    # Average several samples: one reading can land between kernels and look idle.
    SAMPLES=""
    for _ in 1 2 3 4 5; do
      U=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
      SAMPLES="$SAMPLES ${U:-0}"
      sleep 2
    done
    GPU_AVG=$(echo "$SAMPLES" | awk '{s=0;n=0;for(i=1;i<=NF;i++){s+=$i;n++} if(n>0)printf "%d",s/n; else print 0}')
    echo "gpu_avg_util=${GPU_AVG}%"
    if [ "${GPU_AVG:-0}" -lt 20 ] && [ -n "${PROCS:-}" ]; then
      echo "VERDICT: GPU_IDLE_WHILE_TRAINING (avg ${GPU_AVG}%; data-loader or IO stall?)"
    fi

    echo "-- completed metrics --"
    find "$RUN_ROOT" -name metrics.json -printf '%p\n' 2>/dev/null | sort
    echo
  } >> "$OUT"
  sleep "$INTERVAL"
done
