#!/bin/bash
# After a Kunlunxin Megatron run finishes: export stage weights to HF, then
# evaluate base + trained on the held-out test split.
#
#   bash examples/retrieval/finish-kunlunxin-reranker.sh
#
# Megatron keeps one stage{K}.pt per pipeline stage, which the FlagScale
# evaluator cannot read; mrr/export_hf.py merges them into a standard HF
# checkpoint first.  RUN_ROOT must match what run-kunlunxin.sh / the launch used
# so `evaluate_models.sh` finds `<RUN_ROOT>/qwen3_reranker`.
set -uo pipefail
ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
MG=/data2/jiangxuhui/megatron-reranker
RUN_ROOT="${RUN_ROOT:-/data2/jiangxuhui/megatron-reranker/runs/reranker8b-pp2-split}"
STAGES="${STAGES:-$RUN_ROOT/checkpoint}"
HF_OUT="${HF_OUT:-/data2/jiangxuhui/flagos-retrieval-ckpts/kunlunxin-reranker-3epoch/qwen3_reranker}"
PP="${PP:-2}"
MODEL=/flagos-retrieval-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B

cd "$ROOT" || exit 1
source /data2/jiangxuhui/flagos-retrieval-vendor-env.sh
export PYTHONPATH="$MG:$MG/Megatron-LM-FL:$ROOT"
export XMLIR_BMM_DISPATCH_VALUE=0
export TE_FL_SKIP_CUDA=1
export TE_FL_PREFER=vendor
export PYTHONUNBUFFERED=1

echo "===== export $(date '+%F %T') stages=$STAGES pp=$PP -> $HF_OUT ====="
mkdir -p "$HF_OUT"
python -m mrr.export_hf --stages "$STAGES" --pp "$PP" --model "$MODEL" --out "$HF_OUT" || exit 1

# The evaluator expects the model directory directly under RUN_ROOT, and it
# derives `base` from the config's model_path, so only the trained copy needs
# placing.  A symlink keeps the export and the eval path in one place.
EVAL_ROOT=/data2/jiangxuhui/flagos-retrieval-ckpts/kunlunxin-reranker-3epoch
mkdir -p "$EVAL_ROOT"
rm -rf "$EVAL_ROOT/qwen3_reranker"
ln -s "$HF_OUT" "$EVAL_ROOT/qwen3_reranker"

echo "===== evaluate $(date '+%F %T') ====="
export RUN_ROOT="$EVAL_ROOT"
export EVAL_DEVICE=cuda
export PYTHON="${PYTHON:-/data2/jiangxuhui/envs/retrieval-vendor/bin/python}"
export MODELS=qwen3_reranker
export TAGS="base trained"
# 16 candidates per query (1 positive + 15 shared negatives), which is what the
# t2ranking triplet files provide and what the A100/PPU evaluations use.  This is
# the eval candidate count, not the training pair count (conf num_negatives: 1).
export NUM_NEGATIVES="${NUM_NEGATIVES:-15}"
# The 8B reranker at max_length 640 leaves little headroom on 96 GB; the eval
# batch that fits here is smaller than the A100 default.
export BATCH_SIZE="${BATCH_SIZE:-8}"
bash examples/retrieval/evaluate_models.sh || exit 1

echo "===== done $(date '+%F %T') ====="
grep -h "" "$EVAL_ROOT/eval/"*.json 2>/dev/null | head -40
