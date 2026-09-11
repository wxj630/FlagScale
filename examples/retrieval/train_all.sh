#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-${FLAGSCALE_PYTHON_ENV:-/flagos-search-codes/torch-fl-cuda/.venv}/bin/python}"
PROJECT="${WANDB_PROJECT:-flagscale-retrieval}"
RUN_ROOT="${RUN_ROOT:-}"

models=(
  bge_m3
  qwen3_embedding
  qwen3_vl_embedding
  qwen3_reranker
  clip_vit_large
)

for model in "${models[@]}"; do
  echo "===== FlagScale retrieval training: ${model} ====="
  overrides=()
  if [[ -n "$RUN_ROOT" ]]; then
    overrides+=("train.output_dir=${RUN_ROOT}/${model}")
    overrides+=("experiment.exp_dir=${RUN_ROOT}/${model}/hydra")
  fi
  "$PYTHON" -m flagscale.run \
    --config-path examples/retrieval/conf \
    --config-name train \
    action=test \
    train="${model}" \
    +train.wandb_project="${PROJECT}" \
    +train.wandb_name="${model}-3epoch" \
    "${overrides[@]}"
done
