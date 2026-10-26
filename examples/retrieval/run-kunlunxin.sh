#!/bin/bash
# Run one retrieval model on one Kunlunxin card via FlagScale + vendor torch.
#
#   bash run-kunlunxin.sh bge_m3 1        # model, card index
#
# Only the model, parallelism, device and output location are passed. Batch size,
# learning rate, epochs, max_length and the train/test split come from
# conf/train/<model>.yaml so every platform runs the same experiment
# (AGENTS.md section 8).
set -uo pipefail
source /data2/jiangxuhui/flagos-retrieval-vendor-env.sh
# Resolve the repo root from this script's location (examples/retrieval/),
# so the same file works on every host; FLAGSCALE_HOME still overrides.
ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT" || exit 1

model=${1:?model}
card=${2:-1}
export CUDA_VISIBLE_DEVICES="$card"
export XPU_VISIBLE_DEVICES="$card"

# Outputs and scratch on the data disk: the host / is 100% full and anything
# written under it fails with ENOSPC mid-run.
OUT=/data2/jiangxuhui/flagos-retrieval-ckpts/${model}
LOGS=$OUT/logs
export TMPDIR=/data2/jiangxuhui/tmp
export TORCH_EXTENSIONS_DIR=/data2/jiangxuhui/cache/torch_extensions
mkdir -p "$OUT" "$LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

export FLAGOS_USE_FLAGGEMS=false
export PYTHONUNBUFFERED=1

echo "===== $model card=$card start $(date '+%F %T') ====="
$PY -m flagscale.run \
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
