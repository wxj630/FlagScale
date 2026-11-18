#!/usr/bin/env bash
# P800 FSDP2 reranker training, run as many short jobs over pre-cut data chunks.
#
# Why short jobs: this machine dies inside the vendor XCCL stack
# (`kl3_all_gather_raw`, KL_XID_KERNEL_EXCEPTION) after a number of optimizer
# steps that VARIES between runs -- 2,260 on three runs, then 1,220 on
# 2026-11-18 19:01, which that time produced no traceback at all and exited 0
# because the launcher wrapper returned `sync`'s status instead of the
# trainer's. Nothing survives such a death except what is already on disk, so
# the run is split into chunks that each restart the process and get a fresh
# communicator. Rebuilding the process group inside a live process is not
# viable: FSDP2 binds it into every parameter's state.
#
# Why pre-cut files instead of `data.skip_samples`: the skip has to be applied
# after the shuffle buffer (applied before it, a chunk re-read up to 20k rows of
# the previous one -- this happened), and a skip past the end of the stream
# yields nothing rather than wrapping. Reading a different file per job has
# neither failure mode, and `ls` shows exactly which rows ran. Produce them with
# tools/chunk_dataset.py.
#
# Each job:
#   * reads ONLY its own chunk file, so chunks cannot overlap;
#   * warm-starts from the previous chunk's `checkpoint-step-*` (or
#     `checkpoint-epoch-1` if it finished cleanly);
#   * keeps its lr on one global cosine, re-derived per chunk so the schedule
#     runs continuously instead of restarting warmup at every boundary.
#
# Progressive resume: within a chunk, training also snapshots every
# SAVE_STEPS steps. If the job dies mid-chunk, re-running it picks up the
# newest snapshot and continues from that step, so a death costs at most
# SAVE_STEPS steps of work rather than the whole chunk.
#
# Optimizer moments are not saved, so each resume restarts AdamW's moments.
# That is a real, documented compromise: with 300-step snapshots and a total of
# 9,000 steps the moments re-warm roughly 30 times.
#
# Usage:
#   bash run-p800-segments.sh                 # all chunks, in order, resume-aware
#   CHUNK=3 bash run-p800-segments.sh         # just one
#   DRY_RUN=1 bash run-p800-segments.sh       # print the plan, train nothing
set -uo pipefail

REPO=${FLAGSCALE_HOME:-/data2/jiangxuhui/FlagScale}
: "${FLAGSCALE_PYTHON_ENV:?set FLAGSCALE_PYTHON_ENV}"
PY="$FLAGSCALE_PYTHON_ENV/bin/python"
export PATH="$FLAGSCALE_PYTHON_ENV/bin:$PATH"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

MODEL=${MODEL_PATH:-/flagos-retrieval-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B}
# Validation/test always come from the FULL corpus: chunking must not shrink the
# evaluation set, or the reported nDCG stops being comparable to A100/PPU.
EVAL_DATA=${EVAL_DATA:-/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15}
CHUNK_DIR=${CHUNK_DIR:-/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15-chunks}
ROOT_OUT=${OUT:-/data2/jiangxuhui/flagos-retrieval-ckpts/p800-chunks}

BATCH=${BATCH_SIZE:-16}
STEPS_PER_CHUNK=${STEPS_PER_CHUNK:-900}
TOTAL_CHUNKS=${TOTAL_CHUNKS:-10}
SAVE_STEPS=${SAVE_STEPS:-300}
FIRST=${CHUNK:-1}
CARDS=${CARDS:-0,1}
DRY_RUN=${DRY_RUN:-0}
MAX_RETRIES=${MAX_RETRIES:-2}

BASE_LR=${BASE_LR:-1.0e-5}
MIN_LR=${MIN_LR:-1.0e-6}
# No per-chunk warmup: a restart would re-enter warmup and drop the rate to ~0
# at every boundary. The global schedule is carried by lr_at below.
WARMUP=${WARMUP:-0}
TOTAL_STEPS=$(( STEPS_PER_CHUNK * TOTAL_CHUNKS ))

export CUDA_VISIBLE_DEVICES="$CARDS"
export XPU_VISIBLE_DEVICES="$CARDS"
read -ra CARD_LIST <<< "${CARDS//,/ }"
NPROC=${#CARD_LIST[@]}

export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export FLAGSCALE_DEVICE=cuda FS_PLATFORM=cuda FLAGSCALE_AUTOCAST_DEVICE=cuda
export FLAGOS_USE_FLAGGEMS=false WANDB_MODE=disabled WANDB_DISABLED=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Deliberately unset: with it set, every step logged one ASSERT-FAIL per rank
# (0 without it) and it does not affect the XCCL failure.
unset XMLIR_BMM_DISPATCH_VALUE

# XBLAS compiles kernels at first use and caches them under a per-user temp dir.
# Its default is /tmp/xpujitc-temp, created with mode 700 by whichever user got
# there first -- on 2026-11-19 that was a different account on a shared host, so
# every conv/matmul died with
#   Open file /tmp/xpujitc-temp/<pid>/xpu30/0.xpu failed Error:(Permission denied)
# and the process then hung inside the kernel instead of raising, which reads
# like a dead card. Pointing both variables at a directory we own fixes it; the
# failure mode is silent enough that the variables are set unconditionally
# rather than only when the default looks broken.
JIT_DIR=${XBLAS_JIT_DIR:-/data2/jiangxuhui/xblas_jitc}
export XBLAS_JITC_CACHE_PATH="$JIT_DIR/cache"
export XPUJITC_SAVE_TEMP="$JIT_DIR/tmp"
export TMPDIR="${TMPDIR:-$JIT_DIR/tmp}"
mkdir -p "$XBLAS_JITC_CACHE_PATH" "$XPUJITC_SAVE_TEMP"

cd "$REPO"

lr_at() {  # lr_at <global_step> -> cosine value, same shape as build_lr_scheduler
    # Must mirror flagscale/train/train_retrieval.py:build_lr_scheduler exactly,
    # including its `(step + 1) / warmup` warmup form. An earlier version used
    # `step / warm` with WARMUP=200 while passing `warmup_steps=0`, so chunk 1
    # asked for 1e-5/200 = 5e-8 with min_lr 1e-6 and died on the config guard
    # `min_lr <= lr` before a single step.
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

# Newest snapshot inside a chunk directory, by step number.
# `checkpoint-epoch-1` means the chunk finished, so it is reported as the chunk's
# full length rather than as step 0. Returns "<path> <completed_steps>".
latest_ckpt() {
    local dir="$1"
    local full_steps="$2"
    local best="" best_step=-1
    local p step
    for p in "$dir"/checkpoint-step-*; do
        [ -d "$p" ] || continue
        step=$(basename "$p" | sed 's/checkpoint-step-//')
        case "$step" in ''|*[!0-9]*) continue ;; esac
        if [ "$step" -gt "$best_step" ]; then
            best_step=$step
            best="$p"
        fi
    done
    # A finished epoch is the furthest progress there is; never let a stale step
    # snapshot override it.
    if [ -d "$dir/checkpoint-epoch-1" ]; then
        printf '%s %s' "$dir/checkpoint-epoch-1" "$full_steps"
        return
    fi
    if [ -n "$best" ]; then
        printf '%s %s' "$best" "$best_step"
    fi
}

# Global step count already completed for a chunk start.
start_step_for() {  # start_step_for <chunk_index>
    echo $(( ($1 - 1) * STEPS_PER_CHUNK ))
}

last=$(( FIRST + TOTAL_CHUNKS - 1 ))
for chunk in $(seq "$FIRST" "$last"); do
    OUT_DIR="$ROOT_OUT/chunk$chunk"
    CHUNK_FILE=$(printf '%s/chunk-%03d.parquet' "$CHUNK_DIR" $(( chunk - 1 )))
    GLOBAL_FIRST=$(start_step_for "$chunk")

    if [ ! -f "$CHUNK_FILE" ]; then
        echo "chunk$chunk: missing data file $CHUNK_FILE" >&2
        echo "  run: python tools/chunk_dataset.py --data $EVAL_DATA --out $CHUNK_DIR" >&2
        exit 1
    fi

    # Already complete?
    if [ -f "$OUT_DIR/training_complete.json" ] || [ -d "$OUT_DIR/checkpoint-epoch-1" ]; then
        echo "chunk$chunk: already complete, skipping" >&2
        continue
    fi

    # Where to start from, and how far along we already are.
    RESUME_INFO=$(latest_ckpt "$OUT_DIR" "$STEPS_PER_CHUNK")
    RESUME_CKPT=${RESUME_INFO% *}
    RESUME_DONE=${RESUME_INFO#* }
    if [ -n "$RESUME_CKPT" ]; then
        START="$RESUME_CKPT"
        DONE_STEPS=${RESUME_DONE:-0}
        case "$DONE_STEPS" in ''|*[!0-9]*) DONE_STEPS=0 ;; esac
        NOTE="resume $(basename "$RESUME_CKPT") (+$DONE_STEPS)"
    elif [ "$chunk" -eq 1 ]; then
        START="$MODEL"; DONE_STEPS=0; NOTE="from pretrained"
    else
        PREV="$ROOT_OUT/chunk$(( chunk - 1 ))"
        PREV_INFO=$(latest_ckpt "$PREV" "$STEPS_PER_CHUNK")
        START=${PREV_INFO% *}
        DONE_STEPS=0
        NOTE="warm start from chunk$(( chunk - 1 ))"
        if [ -z "$START" ]; then
            echo "chunk$chunk: previous chunk has no checkpoint in $PREV" >&2
            exit 1
        fi
    fi

    REMAIN=$(( STEPS_PER_CHUNK - DONE_STEPS ))
    if [ "$REMAIN" -le 0 ]; then
        echo "chunk$chunk: no steps remaining, skipping" >&2
        continue
    fi
    ROWS=$(( REMAIN * BATCH * NPROC ))
    STEP_NOW=$(( GLOBAL_FIRST + DONE_STEPS ))
    SEG_LR=$(lr_at "$STEP_NOW")
    SEG_END_LR=$(lr_at "$(( GLOBAL_FIRST + STEPS_PER_CHUNK ))")

    echo "===== chunk$chunk $(date '+%F %T') $NOTE | $REMAIN steps from global $STEP_NOW | lr=$SEG_LR->$SEG_END_LR | data=$(basename "$CHUNK_FILE") ====="
    if [ "$DRY_RUN" = "1" ]; then
        echo "  would run: model=$START rows=$ROWS save_steps=$SAVE_STEPS out=$OUT_DIR"
        continue
    fi

    mkdir -p "$OUT_DIR/logs"
    attempt=0
    while :; do
        attempt=$(( attempt + 1 ))
        "$PY" -m flagscale.run \
            --config-path "$REPO/examples/retrieval/conf" --config-name train \
            action=test train=qwen3_reranker train.parallel=fsdp2 train.master_dtype=fp32 \
            experiment.runner.nproc_per_node="$NPROC" \
            train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
            train.hardware.autocast_device=cuda train.hardware.use_flaggems=false \
            train.model.model_path="$START" \
            train.data.path="$CHUNK_FILE" \
            train.data.validation_path="$EVAL_DATA" train.data.test_path="$EVAL_DATA" \
            train.data.validation_format=triplet train.data.test_format=triplet \
            train.batch_size="$BATCH" train.gradient_accumulation_steps=1 \
            train.data.max_samples="$ROWS" train.data.eval_max_samples=64 \
            train.epochs=1 train.log_every=20 train.save_steps="$SAVE_STEPS" \
            train.optimizer.lr="$SEG_LR" train.optimizer.min_lr="$SEG_END_LR" \
            +train.optimizer.lr_decay_steps="$REMAIN" \
            train.optimizer.warmup_steps="$WARMUP" \
            train.output_dir="$OUT_DIR" experiment.exp_dir="$OUT_DIR/logs" \
            hydra.run.dir="$OUT_DIR/logs/hydra" \
            > "$OUT_DIR/train-attempt$attempt.log" 2>&1
        rc=$?
        echo "===== chunk$chunk attempt$attempt exit=$rc $(date '+%F %T') ====="
        if [ "$rc" -eq 0 ]; then
            break
        fi
        # A fresh process gets a fresh communicator, so re-entering is the
        # remedy for the XCCL crash -- but only if a snapshot exists to resume
        # from, otherwise the retry would redo work already lost.
        NEW_INFO=$(latest_ckpt "$OUT_DIR" "$STEPS_PER_CHUNK")
        NEW_CKPT=${NEW_INFO% *}
        NEW_DONE=${NEW_INFO#* }
        if [ -z "$NEW_CKPT" ] || [ "$attempt" -ge "$MAX_RETRIES" ]; then
            echo "chunk$chunk failed (exit $rc) with no usable checkpoint to resume from" >&2
            exit "$rc"
        fi
        START="$NEW_CKPT"
        DONE_STEPS=${NEW_DONE:-0}
        case "$DONE_STEPS" in ''|*[!0-9]*) DONE_STEPS=0 ;; esac
        REMAIN=$(( STEPS_PER_CHUNK - DONE_STEPS ))
        STEP_NOW=$(( GLOBAL_FIRST + DONE_STEPS ))
        SEG_LR=$(lr_at "$STEP_NOW")
        ROWS=$(( REMAIN * BATCH * NPROC ))
        echo "===== chunk$chunk retry $attempt: resuming $(basename "$NEW_CKPT") (+$DONE_STEPS), $REMAIN steps left ====="
    done
done
