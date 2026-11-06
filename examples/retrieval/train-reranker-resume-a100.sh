#!/bin/bash
# Continue the A100 reranker from its own epoch-1 checkpoint, running the
# remaining 2 epochs.
#
# Why "resume" here means "warm start from weights", not a true checkpoint
# resume: save_model() persists parameters only.  There is no optimizer state,
# no LR-schedule position and no RNG state on disk (checked:
# checkpoint-epoch-1/ contains the 5 safetensors shards plus tokenizer/config
# files, and nothing else).  So AdamW moments restart from zero.
#
# The LR is chosen to make that restart harmless: the interrupted run had
# total_steps=14,382 over 3 epochs, so at the end of epoch 1 (step 4,794) its
# cosine schedule sat at 7.86e-6 -- and the log confirms 7.860e-06 on the first
# step of epoch 2.  Starting a fresh 2-epoch cosine from that value reproduces
# the schedule it would have followed, without needing the scheduler state.
#
# Only the starting weights and the schedule are passed on the command line;
# batch size, epochs-per-run, max_length and the data split all come from
# conf/train/qwen3_reranker.yaml.
#
#   bash examples/retrieval/train-reranker-resume-a100.sh
set -uo pipefail
ROOT=${FLAGSCALE_HOME:-/flagos-search-codes/FlagScale}
cd "$ROOT" || exit 1

VN=${FLAGSCALE_PYTHON_ENV:-/flagos-search-codes/sentence-transformers/.venv}
PY="$VN/bin/python"
START=${RERANKER_START:-/flagos-search-ckpts/a100-fp32master-3epoch/qwen3_reranker/checkpoint-epoch-1}
OUT=${RERANKER_OUT:-/flagos-search-ckpts/a100-rr-resume-ep1}
LOG=/flagos-search-codes/logs

mkdir -p "$LOG" "$OUT"
export FLAGSCALE_PYTHON_ENV="$VN"
export PATH="$VN/bin:$PATH"
export PYTHONPATH="$ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT=flagscale-retrieval-a100-fp32master

if [ ! -f "$START/config.json" ]; then
  echo "start checkpoint missing: $START" >&2
  exit 1
fi

echo "[$(date '+%F %T')] resume from $START -> $OUT"
CUDA_VISIBLE_DEVICES=${CARDS:-0,1} "$PY" -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test "train=qwen3_reranker" \
  train.parallel=fsdp2 train.master_dtype=fp32 \
  train.model.model_path="$START" \
  train.epochs=2 \
  train.optimizer.lr=7.86e-6 \
  train.optimizer.min_lr=1.0e-6 \
  train.optimizer.warmup_steps=0 \
  train.log_every=5 \
  experiment.runner.nproc_per_node=2 \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  train.hardware.autocast_device=auto train.hardware.use_flaggems=false \
  train.output_dir="$OUT" experiment.exp_dir="$LOG/exp-qwen3_reranker-resume-ep1" \
  "+train.wandb_name=reranker-resume-from-epoch1" \
  >"$LOG/train-reranker-resume-ep1.log" 2>&1
rc=$?
echo "[$(date '+%F %T')] exit=$rc"
exit "$rc"
