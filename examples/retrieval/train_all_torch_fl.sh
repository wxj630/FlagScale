#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-/flagos-search-codes/torch-fl-cuda/.venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/flagos-search-ckpts/torch-fl-cuda-3epoch-v2}"
PROJECT="${WANDB_PROJECT:-flagscale-retrieval-torch-fl-cuda}"
export FLAGSCALE_PYTHON_ENV="${FLAGSCALE_PYTHON_ENV:-/flagos-search-codes/torch-fl-cuda/.venv}"
export FLAGSCALE_DEVICE=flagos
export FS_PLATFORM=flagos
export FLAGSCALE_DISTRIBUTED_BACKEND=flagos
export FLAGSCALE_AUTOCAST_DEVICE=flagos
export FLAGOS_USE_FLAGGEMS="${FLAGOS_USE_FLAGGEMS:-false}"
# Reduces allocator fragmentation, which matters for the long-sequence text
# fine-tunes that run close to the 80GB card's capacity.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${MODELS:-}" ]]; then
  read -r -a models <<< "${MODELS}"
else
  models=(
    bge_m3
    qwen3_embedding
    qwen3_vl_embedding
    qwen3_reranker
    clip_vit_large
  )
fi

failed=()
for model in "${models[@]}"; do
  echo "===== Torch-FL CUDA retrieval training: ${model} ====="
  if "$PYTHON" -m flagscale.run \
    --config-path examples/retrieval/conf \
    --config-name train \
    action=test \
    train="${model}" \
    train.hardware.device_type=flagos \
    train.hardware.distributed_backend=flagos \
    train.hardware.autocast_device=flagos \
    train.hardware.use_flaggems=false \
    +train.wandb_project="${PROJECT}" \
    +train.wandb_name="${model}-torch-fl-cuda-3epoch" \
    train.output_dir="${RUN_ROOT}/${model}" \
    experiment.exp_dir="${RUN_ROOT}/${model}/hydra"; then
    echo "===== ${model} OK ====="
  else
    echo "===== ${model} FAILED (continuing with the remaining models) ====="
    failed+=("${model}")
  fi
done

if ((${#failed[@]})); then
  echo "===== models that failed: ${failed[*]} ====="
  exit 1
fi
echo "===== all models completed ====="
