#!/usr/bin/env bash
# Segmented P800 FSDP2 reranker training.
#
# Why segments: the run dies deterministically at optimizer step ~2260 inside
# `kl3_all_gather_raw` (KL_XID_KERNEL_EXCEPTION, kernel log), regardless of card
# pair, of ASSERT noise, and of XMLIR_BMM_DISPATCH_VALUE.  That is a cumulative
# failure in the XCCL/device layer, not something a config change fixes.
#
# Rebuilding the process group *in process* is not viable: FSDP2 binds the group
# into every parameter's state and has no supported way to swap it.  Restarting
# the process achieves the same thing safely -- a new process gets a fresh XCCL
# context and a fresh process group -- so each segment stays below the failure
# point and warm-starts the next one from its checkpoint.
#
# Each segment:
#   * max_samples caps rows per rank, so steps = max_samples / batch_size
#   * epochs=1 so the loop ends normally and writes checkpoint-epoch-1
#   * a different shuffle_seed so consecutive segments see different data
#     (same seed would re-train on the same first-N rows every time)
#   * warm starts from the previous segment's weights
#
# Usage:
#   SEGMENT=1 bash run-p800-segments.sh            # first segment
#   SEGMENT=2 bash run-p800-segments.sh            # warm start from segment 1
#   ... or SEGMENTS=4 bash run-p800-segments.sh    # run several back to back
set -uo pipefail

REPO=${FLAGSCALE_HOME:-/data2/jiangxuhui/FlagScale}
: "${FLAGSCALE_PYTHON_ENV:?set FLAGSCALE_PYTHON_ENV}"
PY="$FLAGSCALE_PYTHON_ENV/bin/python"
export PATH="$FLAGSCALE_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL_PATH:-/flagos-retrieval-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B}
DATA=${DATA_PATH:-/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15}
ROOT_OUT=${OUT:-/data2/jiangxuhui/flagos-retrieval-ckpts/p800-seg}

BATCH=${BATCH_SIZE:-16}
STEPS_PER_SEG=${STEPS_PER_SEG:-2000}
SEGMENTS=${SEGMENTS:-1}
FIRST=${SEGMENT:-1}
CARDS=${CARDS:-0,1}

# steps = max_samples / batch_size, and max_samples is per rank.
MAX_SAMPLES=$(( STEPS_PER_SEG * BATCH ))

export CUDA_VISIBLE_DEVICES="$CARDS"
export XPU_VISIBLE_DEVICES="$CARDS"
read -ra CARD_LIST <<< "${CARDS//,/ }"
NPROC=${#CARD_LIST[@]}

export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export FLAGSCALE_DEVICE=cuda FS_PLATFORM=cuda FLAGSCALE_AUTOCAST_DEVICE=cuda
export FLAGOS_USE_FLAGGEMS=false WANDB_MODE=disabled WANDB_DISABLED=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Deliberately NOT setting XMLIR_BMM_DISPATCH_VALUE: it was the source of the
# per-step ASSERT-FAIL spam (0 with it out) and does not affect the 2260 crash.
unset XMLIR_BMM_DISPATCH_VALUE

cd "$REPO"
last=$(( FIRST + SEGMENTS - 1 ))
for seg in $(seq "$FIRST" "$last"); do
    OUT_DIR="$ROOT_OUT/seg$seg"
    if [ -d "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
        echo "seg$seg: $OUT_DIR already exists, skipping" >&2
        continue
    fi
    mkdir -p "$OUT_DIR/logs"

    if [ "$seg" -eq 1 ]; then
        START="$MODEL"
        NOTE="from pretrained"
    else
        START="$ROOT_OUT/seg$(( seg - 1 ))/checkpoint-epoch-1"
        NOTE="warm start from seg$(( seg - 1 ))"
        if [ ! -d "$START" ]; then
            echo "seg$seg: missing $START, run segment $(( seg - 1 )) first" >&2
            exit 1
        fi
    fi

    # Distinct shuffle per segment so the rows differ; the seed also feeds the
    # per-rank/worker offsets.
    SHUFFLE_SEED=$(( 42 + (seg - 1) * 100003 ))

    echo "===== seg$seg start $(date '+%F %T') $NOTE seed=$SHUFFLE_SEED steps=$STEPS_PER_SEG ====="
    "$PY" -m flagscale.run \
        --config-path "$REPO/examples/retrieval/conf" --config-name train \
        action=test train=qwen3_reranker train.parallel=fsdp2 train.master_dtype=fp32 \
        experiment.runner.nproc_per_node="$NPROC" \
        train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
        train.hardware.autocast_device=cuda train.hardware.use_flaggems=false \
        train.model.model_path="$START" \
        train.data.path="$DATA" train.data.validation_path="$DATA" train.data.test_path="$DATA" \
        +train.data.shuffle_seed="$SHUFFLE_SEED" \
        train.batch_size="$BATCH" train.gradient_accumulation_steps=1 \
        train.data.max_samples="$MAX_SAMPLES" train.data.eval_max_samples=64 \
        train.epochs=1 train.log_every=20 \
        train.output_dir="$OUT_DIR" experiment.exp_dir="$OUT_DIR/logs" \
        hydra.run.dir="$OUT_DIR/logs/hydra" \
        > "$OUT_DIR/train.log" 2>&1
    rc=$?
    echo "===== seg$seg exit=$rc $(date '+%F %T') ====="
    if [ "$rc" -ne 0 ]; then
        echo "seg$seg failed (exit $rc); stop here so the failure is visible" >&2
        exit "$rc"
    fi
done
