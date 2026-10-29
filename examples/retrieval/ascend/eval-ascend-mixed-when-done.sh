#!/usr/bin/env bash
# Wait for the BGE/Reranker runs to finish, then evaluate the final checkpoints
# on the full test split with per-query predictions saved.
set -uo pipefail

ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
VENV=${FLAGSCALE_PYTHON_ENV:-/home/jiangxuhui/venvs/flagos-ascend}
BASE=/flagos-retrieval-ckpts/ascend-mixed-20261027
EVAL_DIR="$BASE/eval"
CARD=${ASCEND_RT_VISIBLE_DEVICES:-0}
mkdir -p "$EVAL_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export FLAGSCALE_DEVICE=npu
export FS_PLATFORM=npu
export ASCEND_RT_VISIBLE_DEVICES="$CARD"
export PYTHONPATH="$ROOT"
cd "$ROOT" || exit 1

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
wait_done() {
  local name="$1" dir="$2"
  log "waiting for $name final checkpoint ($dir/training_complete.json)"
  while [ ! -f "$dir/training_complete.json" ]; do sleep 60; done
  log "$name finished"
}

run_bge() {
  log "BGE eval start"
  "$VENV/bin/python" examples/retrieval/evaluate_retrieval.py \
    --model-type bge_m3 \
    --checkpoint "$BASE/bge-m3-gc" \
    --model-path /flagos-retrieval-models/bge-m3/bge-m3 \
    --train-config examples/retrieval/conf/train/bge_m3.yaml \
    --task embedding \
    --data-path /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15 \
    --split test --num-negatives 15 --batch-size 16 --max-length 512 \
    --device npu --save-predictions \
    --output "$EVAL_DIR/bge-m3_test_trained.json" \
    2>&1 | tee "$EVAL_DIR/bge-m3-eval.log"
  log "BGE eval exit=${PIPESTATUS[0]}"
}

run_reranker() {
  log "Reranker eval start (full test, may take hours)"
  "$VENV/bin/python" examples/retrieval/evaluate_retrieval.py \
    --model-type qwen3_reranker \
    --checkpoint "$BASE/qwen3-reranker-8b-4card-bs16-expand" \
    --model-path /flagos-retrieval-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B \
    --train-config examples/retrieval/conf/train/qwen3_reranker.yaml \
    --task reranker \
    --data-path /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15 \
    --split test --num-negatives 15 --batch-size 8 --max-length 640 \
    --device npu --save-predictions \
    --output "$EVAL_DIR/qwen3-reranker-8b_test_trained.json" \
    2>&1 | tee "$EVAL_DIR/qwen3-reranker-8b-eval.log"
  log "Reranker eval exit=${PIPESTATUS[0]}"
}

wait_done "BGE" "$BASE/bge-m3-gc"
run_bge
wait_done "Reranker" "$BASE/qwen3-reranker-8b-4card-bs16-expand"
run_reranker
log "all pending evaluations finished"
