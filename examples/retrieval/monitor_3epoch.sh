#!/usr/bin/env bash
set -u

OUT="${1:-/flagos-search-ckpts/torch-fl-cuda-3epoch-v2-monitor.log}"
INTERVAL="${MONITOR_INTERVAL_SECONDS:-1800}"
RUN_ROOT="${RUN_ROOT:-/flagos-search-ckpts/torch-fl-cuda-3epoch-v2}"

mkdir -p "$(dirname "$OUT")"
while true; do
  {
    date -Is
    echo "== tmux/job processes =="
    ps -eo pid,ppid,etime,%cpu,%mem,cmd | grep -E 'flagscale.run|train_retrieval|torch.distributed.run' | grep -v grep || true
    echo "== GPU =="
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
    echo "== latest training lines =="
    # Prefer the per-rank stdout written under logs/details: the tmux --tee
    # copy in host_0_localhost.output lags the real progress by many minutes.
    LATEST=$(find "$RUN_ROOT" -path '*/attempt_0/*/stdout.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    if [ -z "$LATEST" ]; then
      LATEST=$(find "$RUN_ROOT" -name host_0_localhost.output -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
    fi
    [ -n "$LATEST" ] && tail -n 12 "$LATEST"
    echo "== completed metrics =="
    find "$RUN_ROOT" -name metrics.json -printf '%p\n' 2>/dev/null | sort
    echo
  } >> "$OUT"
  sleep "$INTERVAL"
done
