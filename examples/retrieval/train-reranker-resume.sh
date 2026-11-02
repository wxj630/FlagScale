#!/bin/bash
# Continue the A100 reranker from a valid checkpoint.
#
# Why this exists: the FSDP2 snapshot writer used to persist the adapter's
# state_dict verbatim, so every tensor kept a `backbone.` prefix and arrived
# flattened to 1-D.  HF's from_pretrained reported 399/399 MISSING and silently
# evaluated a randomly initialised model, which is why the "trained" reranker
# looked worse than the pretrained one.  The writer is fixed; the two existing
# snapshots were repaired offline into reranker-epoch{1,2}-fixed, and this run
# starts from epoch 2 because it ranks best (88.5% pairwise vs base 86.5%).
#
# Only the starting weights and the schedule change:
#   * 1 epoch, because epoch 2 already had the best ranking and epoch 3 was
#     where the old run's validation loss climbed furthest;
#   * lr 2e-6 rather than the 1e-5 the 3-epoch run started at, which is roughly
#     where its cosine schedule had decayed to by the end;
#   * a separate output dir so the retained snapshots are not overwritten.
#
#   bash examples/retrieval/train-reranker-resume.sh
set -uo pipefail
ROOT=${FLAGSCALE_HOME:-/flagos-search-codes/FlagScale}
cd "$ROOT" || exit 1

PY=${FLAGSCALE_PYTHON:-/flagos-search-codes/sentence-transformers/.venv/bin/python}
OUT=${RERANKER_OUT:-/flagos-search-ckpts/reranker-resume-ep3}
START=${RERANKER_START:-/flagos-search-ckpts/reranker-epoch2-fixed}
LOG=/flagos-search-codes/logs

mkdir -p "$LOG"
export FLAGSCALE_PYTHON_ENV=/flagos-search-codes/sentence-transformers/.venv
export PYTHONPATH="$ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=flagscale-retrieval-a100-fp32master

if [ ! -f "$START/config.json" ]; then
  echo "start checkpoint missing: $START" >&2
  exit 1
fi

echo "[$(date '+%F %T')] resume from $START -> $OUT"
FLAGSCALE_OUTPUT_DIR="$OUT" CUDA_VISIBLE_DEVICES="${CARDS:-0,1}" "$PY" -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test "train=qwen3_reranker" \
  train.parallel=fsdp2 train.master_dtype=fp32 \
  train.model.model_path="$START" \
  train.epochs=1 \
  train.optimizer.lr=2.0e-6 \
  train.optimizer.min_lr=2.0e-7 \
  train.optimizer.warmup_steps=50 \
  train.log_every=5 \
  experiment.runner.nproc_per_node=2 \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  train.hardware.autocast_device=auto train.hardware.use_flaggems=false \
  experiment.exp_dir="$LOG/exp-qwen3_reranker-resume" \
  +train.wandb_project="$WANDB_PROJECT" \
  "+train.wandb_name=reranker-resume-from-epoch2" \
  >"$LOG/train-reranker-resume.log" 2>&1
echo "[$(date '+%F %T')] exit=$?"
