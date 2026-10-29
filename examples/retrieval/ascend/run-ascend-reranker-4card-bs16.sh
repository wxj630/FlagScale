#!/usr/bin/env bash
# 4-card Ascend FSDP2 reranker, micro-batch 16 x grad-accum 1 (the "bs8" in the old filename was stale).
# Effective batch is 16 * 4 cards * 1 = 64.
set -uo pipefail

ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
VENV=${FLAGSCALE_PYTHON_ENV:-/home/jiangxuhui/venvs/flagos-ascend}
OUT=${FLAGSCALE_OUTPUT_DIR:-/flagos-retrieval-ckpts/ascend-mixed-20261027/qwen3-reranker-8b-4card-bs16-expand}
CARDS=${ASCEND_RT_VISIBLE_DEVICES:-6,7,8,9}
LOGS="$OUT/logs"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
export FLAGSCALE_PYTHON_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT"
export ASCEND_RT_VISIBLE_DEVICES="$CARDS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HCCL_WHITELIST_DISABLE=1
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
# Reduce allocator fragmentation across the many FSDP2 all-gather buffers.
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export TMPDIR="$OUT/tmp"
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
mkdir -p "$OUT" "$LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
cd "$ROOT" || exit 1

echo "===== reranker fsdp2 cards=$CARDS bs=16 ga=1 start $(date '+%F %T') ====="
"$VENV/bin/python" -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=qwen3_reranker \
  train.parallel=fsdp2 train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=4 \
  train.hardware.device_type=npu train.hardware.distributed_backend=hccl \
  train.hardware.autocast_device=npu train.hardware.use_flaggems=false \
  train.batch_size=16 train.gradient_accumulation_steps=1 \
  train.output_dir="$OUT" experiment.exp_dir="$LOGS" hydra.run.dir="$LOGS/hydra" \
  2>&1 | tee "$LOGS/train-qwen3_reranker.log"
echo "===== reranker fsdp2 exit=${PIPESTATUS[0]} end $(date '+%F %T') ====="
