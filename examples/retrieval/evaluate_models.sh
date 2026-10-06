#!/usr/bin/env bash
# Retrieval-quality evaluation for the five models, base vs fine-tuned.
#
# Loss alone does not show whether an encoder retrieves the right document, so
# every model is scored with rank-based metrics (recall@k / MRR). Base and
# fine-tuned checkpoints are always measured with identical settings, because
# comparing different sample counts produced a misleading conclusion earlier in
# this work.
#
# Ranking protocol:
#   * text models  - each query's positive is ranked against its hard negatives
#                    (num_negatives candidates in the pool);
#   * image models - symmetric in-batch image/text retrieval over the test
#                    shards, reported in both directions.
#
# Usage:
#   MODELS="bge_m3 qwen3_reranker" ./evaluate_models.sh
#   SPLIT=validation MAX_SAMPLES=500 ./evaluate_models.sh
#   TAGS=base ./evaluate_models.sh          # skip the fine-tuned pass
#
# Env:
#   MODELS        models to evaluate (default: all five)
#   TAGS          which checkpoints to score (default: "base trained")
#   SPLIT         override the split (default: inferred per task)
#   NUM_NEGATIVES negative candidates for text models (default: 15)
#   MAX_SAMPLES   cap queries/images per run (default: whole split)
#   BATCH_SIZE    override every model's batch (default: per-model below)
#   WAIT_FOR_IDLE set to 1 to block until the device is free (default: 0)
#   RUN_ROOT      where fine-tuned checkpoints live
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-/flagos-search-ckpts/torch-fl-cuda-3epoch-v2}"
EVAL_DIR="${EVAL_DIR:-${RUN_ROOT}/eval}"
MODELS="${MODELS:-bge_m3 qwen3_embedding qwen3_vl_embedding qwen3_reranker clip_vit_large}"
TAGS="${TAGS:-base trained}"
NUM_NEGATIVES="${NUM_NEGATIVES:-15}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-0}"
CONF_DIR="examples/retrieval/conf/train"

export FLAGSCALE_DEVICE=flagos
export FS_PLATFORM=flagos
export FLAGSCALE_DISTRIBUTED_BACKEND=flagos
export FLAGSCALE_AUTOCAST_DEVICE=flagos
# Long-sequence 8B models evaluate close to the card's capacity; fragmentation
# otherwise turns a workable run into an OOM partway through.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$EVAL_DIR"

# Per-model batch. An 8B reranker scoring 16 candidates per query, and an 8B
# vision-language model over image tokens, need far smaller batches than the
# 0.6B text encoders.
default_batch() {
  case "$1" in
    qwen3_reranker) echo 12 ;;
    qwen3_vl_embedding) echo 8 ;;
    clip_vit_large) echo 64 ;;
    *) echo 16 ;;
  esac
}

conf_value() { "$PYTHON" examples/retrieval/conf_value.py "$CONF_DIR/$1.yaml" "$2"; }

if [ "$WAIT_FOR_IDLE" = "1" ]; then
  echo "===== waiting for an idle device at $(date '+%F %T') ====="
  for _ in $(seq 1 11520); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "$used" -lt 3000 ] && [ "$procs" -eq 0 ] && break
    sleep 30
  done
fi

for model in $MODELS; do
  task="$(conf_value "$model" task)"
  max_length="$(conf_value "$model" 'model.max_length')"
  base_path="$(conf_value "$model" model.model_path)"
  batch="${BATCH_SIZE:-$(default_batch "$model")}"

  # Image-text configs score against their dedicated test shards with the "all"
  # split; text configs use the held-out rows of the shared parquet.
  if [ -n "${SPLIT:-}" ]; then
    split="$SPLIT"
  elif [ "$task" = "clip" ] || [ "$task" = "vl_embedding" ]; then
    split="all"
  else
    split="test"
  fi
  if [ "$split" = "all" ]; then
    data="$(conf_value "$model" 'data.test_path' || true)"
  else
    data="$(conf_value "$model" "data.${split}_path" || true)"
  fi
  [ -n "${data:-}" ] || data="$(conf_value "$model" data.path)"

  for tag in $TAGS; do
    case "$tag" in
      base) weights="$base_path" ;;
      *)    weights="${RUN_ROOT}/${model}" ;;
    esac
    if [ ! -d "$weights" ]; then
      echo "----- ${model} [${tag}] SKIPPED (no checkpoint at ${weights}) -----"
      continue
    fi

    echo "----- ${model} [${tag}] split=${split} batch=${batch} $(date '+%F %T') -----"
    args=(
      --model-type "$model"
      --model-path "$weights"
      --checkpoint "$weights"
      # Reuse the training settings: inferring them from the task name once made
      # a correctly fine-tuned reranker score through a random head.
      --train-config "$CONF_DIR/${model}.yaml"
      --task "$task"
      --data-path "$data"
      --split "$split"
      --num-negatives "$NUM_NEGATIVES"
      --batch-size "$batch"
      --max-length "$max_length"
      --device flagos
      --output "${EVAL_DIR}/${model}_${split}_${tag}.json"
    )
    if [ -n "$MAX_SAMPLES" ]; then
      args+=(--max-samples "$MAX_SAMPLES")
    fi
    "$PYTHON" examples/retrieval/evaluate_retrieval.py "${args[@]}" \
      || echo "  (${model} ${tag} failed; any earlier results are kept)"
  done
done

echo "===== evaluation finished at $(date '+%F %T') ====="
echo "results: ${EVAL_DIR}/<model>_<split>_<tag>.json"
