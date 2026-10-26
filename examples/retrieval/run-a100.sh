#!/bin/bash
# Run one retrieval model on one A100 card (bitahub, native CUDA).
#
#   bash run-a100.sh bge_m3 0
#
# Only the model, device and output location are passed; batch size, learning
# rate, epochs, max_length and the train/test split come from
# conf/train/<model>.yaml so every platform runs the same experiment
# (AGENTS.md section 8).
#
# This host has no torch_fl and no vendor plugin: it uses plain native CUDA
# through the sentence-transformers venv, which is what the earlier A100
# baselines used.
set -uo pipefail
# Resolve the repo root from this script's location (examples/retrieval/),
# so the same file works on every host; FLAGSCALE_HOME still overrides.
ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT" || exit 1

model=${1:?model}
card=${2:-0}
export CUDA_VISIBLE_DEVICES="$card"

VN=${FLAGSCALE_PYTHON_ENV:-/flagos-search-codes/sentence-transformers/.venv}
PY="$VN/bin/python"
# FlagScale's `before_start` sources ${FLAGSCALE_PYTHON_ENV}/bin/activate and then
# runs bare `torchrun`. Export it so the wrapper activates the env that actually
# has torchrun: this host's default path points at an incomplete venv.
export FLAGSCALE_PYTHON_ENV="$VN"
export PATH="$VN/bin:$PATH"
export PYTHONPATH="$ROOT"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Outputs under the /flagos-retrieval-* convention (symlinked to the search
# share on this host) so they land on the large disk, not the container overlay.
OUT=/flagos-retrieval-ckpts/a100-fp32master-3epoch/${model}
LOGS=$OUT/logs
export TMPDIR="$OUT/tmp"
mkdir -p "$OUT" "$LOGS" "$TMPDIR"

echo "===== $model card=$card start $(date '+%F %T') ====="
"$PY" -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test "train=$model" \
  train.parallel=ddp train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=1 \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  train.hardware.autocast_device=auto train.hardware.use_flaggems=false \
  train.output_dir="$OUT" experiment.exp_dir="$LOGS" \
  hydra.run.dir="$LOGS/hydra" \
  > "$LOGS/train-$model.log" 2>&1
rc=$?
echo "===== $model card=$card exit=$rc end $(date '+%F %T') ====="
exit "$rc"
