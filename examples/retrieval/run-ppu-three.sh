#!/bin/bash
# Run the three retrieval models on PPU, each in its own container and card.
#
# Every semantic setting (batch size, epochs, lr, max_length, train/test split)
# comes from conf/train/<model>.yaml, unchanged across platforms -- see
# AGENTS.md section 8. This script only decides *where* to run, not *what*.
#
#   bash /workspace/run-ppu-three.sh          # all three
#   bash /workspace/run-ppu-three.sh bge_m3   # one
set -uo pipefail
# Same as the other runners: resolve the repo from the script location so the
# file is host-independent. Inside the container the repo is /workspace/FlagScale.
ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT" || exit 1

export FLAGSCALE_DEVICE=flagos FS_PLATFORM=flagos
export FLAGSCALE_DISTRIBUTED_BACKEND=flagos FLAGSCALE_AUTOCAST_DEVICE=flagos
export FLAGOS_USE_FLAGGEMS=false
export GEMS_VENDOR=thead
export NCCL_SOCKET_IFNAME=bond0
export PYTHONPATH=/workspace/FlagScale
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# Outputs go under the shared /flagos-retrieval-ckpts convention (a symlink to
# the data disk inside the container). The container's / is a small read-only
# overlay: a full dataset checkpoint there fails mid-save with ENOSPC, which is
# what killed an earlier attempt.
CKPT=/flagos-retrieval-ckpts/ppu-fp32master-3epoch
mkdir -p "$CKPT"

# Python's tempfile and the torch extension cache default to /tmp, which sits on
# the small root filesystem and runs out mid-run ("No space left on device"
# while saving). Point both at the data disk.
export TMPDIR="$CKPT/tmp"
export TORCH_EXTENSIONS_DIR="$CKPT/torch_extensions"
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

train_one() {
  local model=$1 parallel=$2 nproc=$3
  # Per-model log dir: a shared one mixes all three models into one file.
  LOGS="$CKPT/$model/logs"
  mkdir -p "$LOGS"
  echo "===== $model parallel=$parallel nproc=$nproc start $(date '+%F %T') ====="
  python -m flagscale.run \
    --config-path examples/retrieval/conf --config-name train \
    action=test "train=$model" \
    train.parallel="$parallel" train.master_dtype=fp32 \
    experiment.runner.nproc_per_node="$nproc" \
    train.hardware.device_type=flagos train.hardware.distributed_backend=flagos \
    train.hardware.autocast_device=flagos train.hardware.use_flaggems=false \
    train.output_dir="$CKPT/$model" experiment.exp_dir="$LOGS/exp-$model" \
    hydra.run.dir="$LOGS/exp-$model/hydra" \
    > "$LOGS/train-$model.log" 2>&1
  local rc=$?
  echo "===== $model exit=$rc end $(date '+%F %T') ====="
  return "$rc"
}

case "${1:-all}" in
  bge_m3)         train_one bge_m3 ddp 1 ;;
  clip_vit_large) train_one clip_vit_large ddp 1 ;;
  qwen3_reranker) train_one qwen3_reranker fsdp2 2 ;;
esac
