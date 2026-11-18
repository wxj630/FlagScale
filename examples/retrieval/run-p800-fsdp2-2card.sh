#!/usr/bin/env bash
# Qwen3-Reranker-8B FSDP2 on 2 P800 cards, same recipe as A100/PPU.
#
# Only the card count, device and output paths are overridden here.  batch_size,
# lr, epochs, max_length and the data splits come from
# conf/train/qwen3_reranker.yaml so this run is comparable with the A100 and PPU
# acceptance numbers (AGENTS.md section 8).
#
# Usage:
#   FLAGSCALE_PYTHON_ENV=/path/to/venv CARDS=2,3 bash run-p800-fsdp2-2card.sh
#   ... bash run-p800-fsdp2-2card.sh train.data.max_samples=64   # smoke
set -eo pipefail

REPO=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
: "${FLAGSCALE_PYTHON_ENV:?set FLAGSCALE_PYTHON_ENV to the vendor python root}"
PY="$FLAGSCALE_PYTHON_ENV/bin/python"
export PATH="$FLAGSCALE_PYTHON_ENV/bin:$PATH"

MODEL=${MODEL_PATH:-/flagos-retrieval-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B}
DATA=${DATA_PATH:-/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15}
OUT=${OUT:-/data2/jiangxuhui/flagos-retrieval-ckpts/p800-fsdp2-2card}
CARDS=${CARDS:-3,7}

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$CARDS"
export XPU_VISIBLE_DEVICES="$CARDS"
read -ra CARD_LIST <<< "${CARDS//,/ }"
NPROC=${#CARD_LIST[@]}

export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export FLAGSCALE_DEVICE=cuda FS_PLATFORM=cuda FLAGSCALE_AUTOCAST_DEVICE=cuda
export FLAGOS_USE_FLAGGEMS=false WANDB_MODE=disabled WANDB_DISABLED=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# P800 XBLAS picks a bmm kernel that fails to JIT; 0 selects a working one
# (~2x faster at seq 640 as well).  Required for seq >= 288 backwards.
export XMLIR_BMM_DISPATCH_VALUE=0

echo "===== reranker FSDP2 cards=$CARDS nproc=$NPROC out=$OUT start $(date '+%F %T') ====="
mkdir -p "$OUT/logs"

# Same entry point as the A100 and PPU acceptance runs (flagscale.run), so the
# only differences from those numbers are the platform and the card count.
exec "$PY" -m flagscale.run \
    --config-path "$REPO/examples/retrieval/conf" --config-name train \
    action=test train=qwen3_reranker train.parallel=fsdp2 train.master_dtype=fp32 \
    experiment.runner.nproc_per_node="$NPROC" \
    train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
    train.hardware.autocast_device=cuda train.hardware.use_flaggems=false \
    train.model.model_path="$MODEL" \
    train.data.path="$DATA" train.data.validation_path="$DATA" train.data.test_path="$DATA" \
    train.output_dir="$OUT" experiment.exp_dir="$OUT/logs" hydra.run.dir="$OUT/logs/hydra" \
    "$@" > "$OUT/train.log" 2>&1
