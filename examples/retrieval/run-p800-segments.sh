#!/usr/bin/env bash
# Segmented P800 FSDP2 reranker training over DISJOINT data chunks.
#
# Why segments: the run dies inside XCCL (`kl3_all_gather_raw`,
# KL_XID_KERNEL_EXCEPTION in the kernel log) after roughly 2260 optimizer steps.
# Reproduced three times at the same step, independent of card pair, of
# XMLIR_BMM_DISPATCH_VALUE, and of ASSERT noise.  Rebuilding the process group
# inside a live process is not viable -- FSDP2 binds it into every parameter's
# state -- so each segment is its own process, which gets a fresh communicator.
#
# Segmentation must walk *different* rows each time.  Changing the shuffle seed
# would only reshuffle the same corpus and re-expose most of the rows already
# seen, so this uses `data.skip_samples` instead: segment k skips the rows
# earlier segments consumed, keeping the split boundaries and shuffle seed
# fixed.  Verified disjoint and reproducible.
#
# The lr is re-derived per segment so the whole run follows one cosine schedule
# over TOTAL_STEPS, instead of restarting warmup at every segment.  Optimizer
# moments still restart at each boundary (save_model stores weights only).
#
# Usage:
#   bash run-p800-segments.sh                    # all six segments, in order
#   SEGMENT=3 bash run-p800-segments.sh          # just one
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
ROWS_PER_SEG=${ROWS_PER_SEG:-28800}   # per-rank rows; 28800/16 = 1800 steps, x6 covers the split
TOTAL_SEGMENTS=${TOTAL_SEGMENTS:-6}
FIRST=${SEGMENT:-1}
CARDS=${CARDS:-0,1}

BASE_LR=${BASE_LR:-1.0e-5}
MIN_LR=${MIN_LR:-1.0e-6}
# No per-segment warmup: every segment restart would re-enter warmup and drop
# the rate to ~0 at each boundary. The global schedule is carried by lr_at.
WARMUP=${WARMUP:-0}
TOTAL_STEPS=$(( ROWS_PER_SEG / BATCH * TOTAL_SEGMENTS ))

export CUDA_VISIBLE_DEVICES="$CARDS"
export XPU_VISIBLE_DEVICES="$CARDS"
read -ra CARD_LIST <<< "${CARDS//,/ }"
NPROC=${#CARD_LIST[@]}

export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export FLAGSCALE_DEVICE=cuda FS_PLATFORM=cuda FLAGSCALE_AUTOCAST_DEVICE=cuda
export FLAGOS_USE_FLAGGEMS=false WANDB_MODE=disabled WANDB_DISABLED=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Deliberately unset: with it set, every step logged one ASSERT-FAIL per rank
# (0 without it) and it does not affect the 2260-step XCCL failure.
unset XMLIR_BMM_DISPATCH_VALUE

cd "$REPO"

lr_at() {  # lr_at <global_step> -> cosine value, same shape as build_lr_scheduler
    # Must mirror flagscale/train/train_retrieval.py:build_lr_scheduler exactly,
    # including its `(step + 1) / warmup` warmup form. It previously used
    # `step / warm` with the script's WARMUP=200 while passing
    # `warmup_steps=0` to the trainer, so segment 1 started at 1e-5/200 = 5e-8
    # -- below the segment floor -- and the run died on the config guard
    # `min_lr <= lr` before a single step. WARMUP here must equal what is passed
    # below; the trainer owns the schedule, this only picks each segment's start.
    "$PY" - "$1" "$TOTAL_STEPS" "$BASE_LR" "$MIN_LR" "$WARMUP" <<'PY'
import math, sys
step, total = int(sys.argv[1]), int(sys.argv[2])
base, floor_, warm = float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
step = max(0, min(step, total))
if warm and step < warm:
    print(f"{base * (step + 1) / warm:.6e}")
else:
    prog = min(1.0, (step - warm) / max(1, total - warm))
    print(f"{floor_ + (base - floor_) * 0.5 * (1 + math.cos(math.pi * prog)):.6e}")
PY
}

last=$(( FIRST + TOTAL_SEGMENTS - 1 ))
for seg in $(seq "$FIRST" "$last"); do
    OUT_DIR="$ROOT_OUT/seg$seg"
    if [ -d "$OUT_DIR/checkpoint-epoch-1" ]; then
        echo "seg$seg: checkpoint exists, skipping" >&2
        continue
    fi
    rm -rf "$OUT_DIR"; mkdir -p "$OUT_DIR/logs"

    if [ "$seg" -eq 1 ]; then
        START="$MODEL"; NOTE="from pretrained"
    else
        START="$ROOT_OUT/seg$(( seg - 1 ))/checkpoint-epoch-1"
        NOTE="warm start from seg$(( seg - 1 ))"
        [ -d "$START" ] || { echo "seg$seg: missing $START" >&2; exit 1; }
    fi

    STEPS_PER_SEG=$(( ROWS_PER_SEG / BATCH ))
    SKIP=$(( (seg - 1) * ROWS_PER_SEG ))
    FIRST_STEP=$(( (seg - 1) * STEPS_PER_SEG ))
    SEG_LR=$(lr_at "$FIRST_STEP")
    SEG_END_LR=$(lr_at "$(( FIRST_STEP + STEPS_PER_SEG ))")
    # The trainer owns the in-segment schedule and needs `lr_decay_steps` to
    # have one at all: with the total unknown it holds the rate at base_lr for
    # the whole segment (that is what happened on the first attempt, so the
    # segments ran at a constant rate instead of following the cosine).
    echo "===== seg$seg $(date '+%F %T') $NOTE skip=$SKIP rows=$ROWS_PER_SEG lr=$SEG_LR->$SEG_END_LR ====="
    "$PY" -m flagscale.run \
        --config-path "$REPO/examples/retrieval/conf" --config-name train \
        action=test train=qwen3_reranker train.parallel=fsdp2 train.master_dtype=fp32 \
        experiment.runner.nproc_per_node="$NPROC" \
        train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
        train.hardware.autocast_device=cuda train.hardware.use_flaggems=false \
        train.model.model_path="$START" \
        train.data.path="$DATA" train.data.validation_path="$DATA" train.data.test_path="$DATA" \
        +train.data.skip_samples="$SKIP" \
        train.batch_size="$BATCH" train.gradient_accumulation_steps=1 \
        train.data.max_samples="$ROWS_PER_SEG" train.data.eval_max_samples=64 \
        train.epochs=1 train.log_every=20 \
        train.optimizer.lr="$SEG_LR" train.optimizer.min_lr="$SEG_END_LR" \
        +train.optimizer.lr_decay_steps="$STEPS_PER_SEG" \
        train.optimizer.warmup_steps="$WARMUP" \
        train.output_dir="$OUT_DIR" experiment.exp_dir="$OUT_DIR/logs" \
        hydra.run.dir="$OUT_DIR/logs/hydra" \
        > "$OUT_DIR/train.log" 2>&1
    rc=$?
    echo "===== seg$seg exit=$rc $(date '+%F %T') ====="
    if [ "$rc" -ne 0 ]; then
        echo "seg$seg failed (exit $rc)" >&2
        exit "$rc"
    fi
done
