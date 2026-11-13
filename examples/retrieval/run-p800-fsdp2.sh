#!/usr/bin/env bash
# Eight-card P800 FSDP2 smoke: real 8B weights, train/eval, and HF checkpoints.
# Set FLAGSCALE_PYTHON_ENV to an isolated vendor Python environment.
# Extra arguments are Hydra overrides, e.g. train.data.max_samples=32.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
ROOT=${RETRIEVAL_ROOT:-$(dirname "$REPO")}
: "${FLAGSCALE_PYTHON_ENV:?Set FLAGSCALE_PYTHON_ENV to an isolated XPytorch environment}"
PY="$FLAGSCALE_PYTHON_ENV/bin/python"
export PATH="$FLAGSCALE_PYTHON_ENV/bin:$PATH"
export CONDA_PREFIX="$FLAGSCALE_PYTHON_ENV"
if [[ -d "$FLAGSCALE_PYTHON_ENV/xcudart/lib" ]]; then
    export LD_LIBRARY_PATH="$FLAGSCALE_PYTHON_ENV/xcudart/lib:$FLAGSCALE_PYTHON_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=${CARDS:-0,1,2,3,4,5,6,7}
export XPU_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
IFS=, read -r -a CARD_LIST <<< "$CUDA_VISIBLE_DEVICES"
export P800_WORLD_SIZE=${#CARD_LIST[@]}
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export FLAGSCALE_DEVICE=cuda FS_PLATFORM=cuda FLAGSCALE_AUTOCAST_DEVICE=cuda
export FLAGOS_USE_FLAGGEMS=false WANDB_MODE=disabled WANDB_DISABLED=true
export XMLIR_BMM_DISPATCH_VALUE=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export P800_MODEL=${MODEL_PATH:-$ROOT/flagscale-p800-assets/models/Qwen3-Reranker-8B}
export P800_DATA=${DATA_PATH:-$ROOT/flagscale-p800-assets/datasets/t2ranking/triplet-15}
export P800_OUT=${OUT:-$ROOT/flagscale-p800-runs/fsdp2-fp32-smoke-$(date +%Y%m%d-%H%M%S)}
export P800_REPO=$REPO
# Refuse to overwrite an earlier experiment's configuration/checkpoint.
mkdir -p "$(dirname "$P800_OUT")"
mkdir "$P800_OUT"
cd "$REPO"

"$PY" -u - "$@" <<'PY'
import os
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

out = os.environ["P800_OUT"]
data = os.environ["P800_DATA"]
with initialize_config_dir(config_dir=os.environ["P800_REPO"] + "/examples/retrieval/conf", version_base=None):
    cfg = compose(config_name="train", overrides=[
        "train=qwen3_reranker", "train.parallel=fsdp2", "train.master_dtype=fp32",
        "train.model.torch_dtype=bf16", "train.model.max_length=640",
        "train.model.model_path=" + os.environ["P800_MODEL"],
        "train.data.path=" + data, "train.data.validation_path=" + data,
        "train.data.test_path=" + data,
        "train.batch_size=1", "train.gradient_accumulation_steps=1", "train.epochs=1",
        "train.log_every=1", "train.data.max_samples=16", "train.data.eval_max_samples=4",
        "train.data.shuffle_buffer=128", "train.data.length_sort_window=16",
        "train.optimizer.lr=1e-7", "train.optimizer.min_lr=1e-8",
        "train.optimizer.warmup_steps=2", "+train.optimizer.lr_decay_steps=16",
        "train.hardware.device_type=cuda", "train.hardware.autocast_device=cuda",
        "train.hardware.use_flaggems=false", "train.output_dir=" + out,
        "experiment.exp_dir=" + out,
        "experiment.runner.nproc_per_node=" + os.environ["P800_WORLD_SIZE"],
        "experiment.envs.WANDB_MODE=disabled",
    ] + sys.argv[1:])
OmegaConf.save(cfg, out + "/config.yaml", resolve=True)
print("RUN_DIR=" + out, flush=True)
PY
# Call the native trainer directly so its failure is the launcher's exit status.
exec "$PY" -u -m torch.distributed.run --standalone --nproc_per_node="$P800_WORLD_SIZE" \
    -m flagscale.train.train_retrieval --config-file "$P800_OUT/config.yaml"
