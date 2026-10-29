#!/usr/bin/env bash
# BGE-M3 on one Ascend chip with fp32 master weights + bf16 autocast.
# batch_size (10) and gradient_accumulation_steps (4) are unchanged; activation
# checkpointing is enabled only to fit the 64GB HBM (fp32 master + AdamW).
set -uo pipefail

ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
VENV=${FLAGSCALE_PYTHON_ENV:-/home/jiangxuhui/venvs/flagos-ascend}
OUT=${FLAGSCALE_OUTPUT_DIR:-/flagos-retrieval-ckpts/ascend-mixed-20261027/bge-m3-gc}
CARD=${ASCEND_RT_VISIBLE_DEVICES:-2}
LOGS="$OUT/logs"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export FLAGSCALE_PYTHON_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT"
export ASCEND_RT_VISIBLE_DEVICES="$CARD"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HCCL_WHITELIST_DISABLE=1
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export TMPDIR="$OUT/tmp"
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
mkdir -p "$OUT" "$LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
cd "$ROOT" || exit 1

echo "===== bge_m3 gc cards=$CARD start $(date '+%F %T') ====="
"$VENV/bin/python" -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=bge_m3 \
  train.parallel=ddp train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=1 \
  train.hardware.device_type=npu train.hardware.distributed_backend=hccl \
  train.hardware.autocast_device=npu train.hardware.use_flaggems=false \
  +train.model.gradient_checkpointing=true \
  train.output_dir="$OUT" experiment.exp_dir="$LOGS" hydra.run.dir="$LOGS/hydra" \
  2>&1 | tee "$LOGS/train-bge_m3.log"
echo "===== bge_m3 gc exit=${PIPESTATUS[0]} end $(date '+%F %T') ====="
