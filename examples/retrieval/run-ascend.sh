#!/bin/bash
# Run one retrieval model on Ascend NPU (torch_npu + HCCL).
#
#   bash run-ascend.sh bge_m3                 # card 0
#   bash run-ascend.sh clip_vit_large 1       # card 1
#   bash run-ascend.sh qwen3_reranker 0 "0,1" # 2 cards, FSDP2
#
# Only the model, the cards and the output location are passed.  Batch size,
# learning rate, epochs, max_length and the data split come from
# conf/train/<model>.yaml, so an Ascend run is comparable with the A100 / PPU /
# Kunlunxin ones (AGENTS.md section 8).
#
# Unlike PPU this platform does NOT need torch_fl: torch_npu registers the `npu`
# device itself, which is what flagscale/platforms/platform_npu.py drives
# (`device_type=npu`, backend `hccl`).  Keeping Torch-FL out is deliberate --
# fewer layers to debug on a new chip.
#
# Environment this script expects (all overridable):
#   ASCEND_HOME         /usr/local/Ascend  (set_env.sh lives in ascend-toolkit/)
#   FLAGSCALE_PYTHON_ENV  the venv/conda env that has torch, torch_npu, omegaconf
set -uo pipefail
ROOT=${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$ROOT" || exit 1

model=${1:?usage: run-ascend.sh <model> [cards] [card-list]}
cards=${2:-0}
card_list=${3:-$cards}
export ASCEND_RT_VISIBLE_DEVICES="$card_list"

# torch_npu requires the CANN environment.  Source it if the caller has not.
if [ -z "${ASCEND_HOME:-}" ] && [ -d /usr/local/Ascend ]; then
  export ASCEND_HOME=/usr/local/Ascend
fi
if [ -n "${ASCEND_HOME:-}" ] && [ -z "${ASCEND_TOOLKIT_INCLUDE:-}" ]; then
  # shellcheck disable=SC1091
  [ -f "$ASCEND_HOME/ascend-toolkit/set_env.sh" ] && source "$ASCEND_HOME/ascend-toolkit/set_env.sh"
fi

VN=${FLAGSCALE_PYTHON_ENV:-$ROOT/.venv}
PY="$VN/bin/python"
if [ ! -x "$PY" ]; then
  echo "no python at $PY; set FLAGSCALE_PYTHON_ENV to the env holding torch_npu" >&2
  exit 1
fi
# FlagScale's `before_start` sources this env and then runs bare `torchrun`.
export FLAGSCALE_PYTHON_ENV="$VN"
export PATH="$VN/bin:$PATH"
export PYTHONPATH="$ROOT"

# Ascend uses its own allocator; PYTORCH_ALLOC_CONF/expandable_segments is a
# CUDA-allocator knob and is left unset here (setting it only produces a warning).
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HCCL_WHITELIST_DISABLE=1
# Keep the collective log short; raise to INFO when debugging HCCL hangs.
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"

OUT=${FLAGSCALE_OUTPUT_DIR:-/flagos-retrieval-ckpts/ascend-fp32master-3epoch/${model}}
LOGS=$OUT/logs
# The repo may live on a small root partition on these boxes; temp and the torch
# extension cache go next to the checkpoints instead.
export TMPDIR="$OUT/tmp"
export TORCH_EXTENSIONS_DIR="$OUT/torch_extensions"
mkdir -p "$OUT" "$LOGS" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"

# Same reasoning as the other runners: the 8B reranker needs 131 GB of optimizer
# state (8.19B x 16 bytes) and cannot fit one 64 GB NPU, so it shards with FSDP2.
# bge/clip stay on single-card DDP so their numbers match the existing baselines.
PARALLEL=ddp
NPROC=1
if [ "${model}" = "qwen3_reranker" ]; then
  PARALLEL=fsdp2
  NPROC=${RERANKER_CARDS:-2}
fi

echo "===== $model cards=$card_list parallel=$PARALLEL nproc=$NPROC start $(date '+%F %T') ====="
"$PY" -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test "train=$model" \
  train.parallel=$PARALLEL train.master_dtype=fp32 \
  experiment.runner.nproc_per_node="$NPROC" \
  train.hardware.device_type=npu train.hardware.distributed_backend=hccl \
  train.hardware.autocast_device=npu train.hardware.use_flaggems=false \
  train.output_dir="$OUT" experiment.exp_dir="$LOGS" \
  hydra.run.dir="$LOGS/hydra" \
  2>&1 | tee "$LOGS/train-$model.log"
rc=${PIPESTATUS[0]}
echo "===== $model exit=$rc end $(date '+%F %T') ====="
exit "$rc"
