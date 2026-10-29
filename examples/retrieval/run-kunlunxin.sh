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

# Re-assert after sourcing the vendor environment.  That file used to assign
# CUDA_VISIBLE_DEVICES unconditionally (XPU_CARD defaults to 2), which silently
# turned a 2-card run into a 2-ranks-on-one-card run.  The file is fixed now,
# but the vendor env is outside this repo and can drift again.
_assert_cards() {
  if [ "${CUDA_VISIBLE_DEVICES:-}" != "$card" ]; then
    echo "CUDA_VISIBLE_DEVICES was overridden to '${CUDA_VISIBLE_DEVICES:-}' (wanted '$card')" >&2
    echo "something in the sourced environment is reassigning it" >&2
    return 1
  fi
}
_assert_cards || exit 1


VN=${FLAGSCALE_PYTHON_ENV:-/data2/jiangxuhui/envs/retrieval-vendor}
PY="$VN/bin/python"

# Megatron-LM-FL is not installed as a package on these hosts; it is a git clone
# that the trainer imports as `megatron.core`.  Point PYTHONPATH at it only for
# the megatron backend so the ddp/fsdp2 runs keep their original import set.
MEGATRON_HOME=${FLAGSCALE_MEGATRON_HOME:-/data2/jiangxuhui/megatron-reranker/Megatron-LM-FL}

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
# Both backends go through `flagscale.run`: it renders the hydra config and
# launches torchrun with the requested nproc, so the Megatron route gets the
# same config system, exp directory and logging as DDP/FSDP2.
#
# The reranker takes EXTRA args for two P800-specific reasons:
#   * `parallel=megatron`: P800's CUDA shim has no `expandable_segments`, so
#     FSDP2 cannot allocate the 2.31 GiB contiguous all-gather buffer for
#     `lm_head` and degrades to bf16 master weights.  Megatron's pipeline
#     parallel never materialises the whole model on one rank, and its
#     distributed optimizer still shards the fp32 state (16 bytes/param).
#   * `skip_nonfinite`: fp32-master training diverges here without it.
EXTRA=()
PARALLEL=ddp
NPROC=1
if [ "$model" = "qwen3_reranker" ]; then
  PARALLEL=megatron
  NPROC=$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
  STAGE_DIR=${FLAGSCALE_STAGE_DIR:-$ROOT/../megatron-reranker/work/pp${NPROC}-8b}
  if [ ! -f "$STAGE_DIR/stage0.pt" ]; then
    echo "missing $STAGE_DIR/stage0.pt" >&2
    echo "convert once with:" >&2
    echo "  python -m flagscale.models.retrieval.qwen3_reranker_megatron convert \\" >&2
    echo "      --model /flagos-retrieval-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B \\" >&2
    echo "      --pp $NPROC --out $STAGE_DIR" >&2
    exit 1
  fi
  EXTRA+=(
    "train.pipeline_model_parallel_size=$NPROC"
    "train.stage_dir=$STAGE_DIR"
    "train.skip_nonfinite=true"
  )
  if [ ! -d "$MEGATRON_HOME/megatron/core" ]; then
    echo "missing megatron.core under $MEGATRON_HOME" >&2
    echo "set FLAGSCALE_MEGATRON_HOME to your Megatron-LM-FL clone" >&2
    exit 1
  fi
  export PYTHONPATH="$MEGATRON_HOME:$ROOT:${PYTHONPATH:-}"
fi

$PY -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test "train=$model" \
  train.parallel=$PARALLEL train.master_dtype=fp32 \
  experiment.runner.nproc_per_node="$NPROC" \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  train.hardware.autocast_device=auto train.hardware.use_flaggems=false \
  "${EXTRA[@]}" \
  train.output_dir="$OUT" experiment.exp_dir="$LOGS" \
  hydra.run.dir="$LOGS/hydra" \
  > "$LOGS/train-$model.log" 2>&1
rc=$?
echo "===== $model card=$card exit=$rc end $(date '+%F %T') ====="
exit "$rc"
