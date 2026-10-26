# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""Distributed contrastive/ranking training for local retrieval models.

This entrypoint is intentionally backend-agnostic: FlagScale launches it through
the ``native`` training backend, while model adapters, datasets, and losses live
in their corresponding FlagScale modules.  This file wires those modules into
the device/DDP lifecycle and epoch loop for the five local checkpoints used by
the retrieval example:

* Native Transformers bi-encoders (BGE, Qwen3 Embedding and Qwen3-VL Embedding)
* Qwen3-Reranker CrossEncoder
* CLIP image/text contrastive training on COCO WebDataset shards

The script accepts the Hydra-generated configuration produced by FlagScale via
``--config-file=/path/to/config.yaml``.  It also runs directly with a regular
YAML file, which makes small smoke tests easy to run without FlagScale.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

# torch-fl must claim PyTorch's PrivateUse1 slot before the first flagos tensor
# is created.  Keep the import opt-in so the CUDA environment has no dependency
# on a vendor plugin.  The FlagScale config exports FLAGSCALE_DEVICE before
# torchrun starts this module.
_REQUESTED_DEVICE = os.environ.get(
    "FLAGSCALE_DEVICE", os.environ.get("FS_PLATFORM", "auto")
).strip().lower()
if _REQUESTED_DEVICE in {"flagos", "privateuseone"}:
    try:
        import torch_fl  # noqa: F401
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - vendor envs
        raise RuntimeError(
            "FLAGSCALE_DEVICE=flagos requires a matching torch-fl installation. "
            "Install torch-fl in a PyTorch 2.10.x/vendor environment; do not mix it "
            "with the CUDA-only training environment."
        ) from exc

import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from flagscale.models.retrieval import build_retrieval_model, unwrap_model
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.utils import parallel as parallel_utils

try:
    from flagscale.platforms import get_platform
except ImportError:  # pragma: no cover - direct standalone invocation
    get_platform = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", required=True, help="Hydra-generated or standalone YAML")
    return parser.parse_args()


def rank_zero() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(message: str) -> None:
    if rank_zero():
        print(message, flush=True)


def init_distributed(
    hardware: DictConfig | None = None,
) -> tuple[int, int, torch.device, Any]:
    """Initialize torchrun's process group, or fall back to a single process."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    hardware = hardware or {}
    requested = str(
        hardware.get("device_type", os.environ.get("FS_PLATFORM", os.environ.get("FLAGSCALE_DEVICE", "auto")))
    ).lower()
    platform = None
    if get_platform is not None:
        try:
            platform = get_platform()
        except (RuntimeError, ValueError):
            platform = None

    if requested in {"privateuseone", "flagos"}:
        if platform is None or platform.name() != "flagos":
            raise RuntimeError(
                "device_type=flagos was requested, but a Torch-FL platform is "
                "unavailable. Import torch_fl before torch and use its matching "
                "PyTorch 2.10.x environment."
            )
    elif requested not in {"auto", "", "cuda"}:
        if platform is None or platform.name() != requested:
            raise RuntimeError(f"Requested platform {requested!r} is unavailable")
    elif requested == "cuda" and platform is not None and platform.name() != "cuda":
        raise RuntimeError(
            "device_type=cuda was requested, but the active runtime exposes "
            f"{platform.name()!r}. Use device_type=flagos with Torch-FL or a "
            "native CUDA PyTorch environment."
        )
    elif requested in {"auto", ""} and platform is None:
        # Preserve the standalone CPU fallback used by small unit tests.
        device = torch.device("cpu")
        device_kind = "cpu"
    if platform is not None and (requested not in {"auto", ""} or platform.is_available()):
        platform.set_device(local_rank)
        device = platform.device(local_rank)
        device_kind = platform.name()
    elif "device" not in locals():
        device = torch.device("cpu")
        device_kind = "cpu"
    if world_size > 1 and not dist.is_initialized():
        backend = str(
            hardware.get("distributed_backend", os.environ.get("FLAGSCALE_DISTRIBUTED_BACKEND", "auto"))
        ).lower()
        if backend == "auto":
            backend = platform.dist_backend() if platform is not None else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return rank, world_size, device, platform


def finish_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _as_bool(value: Any) -> bool:
    """Parse a config flag that may arrive as a string from ``oc.env``.

    ``bool("false")`` is True, so a config written as
    ``deterministic: ${oc.env:FLAGSCALE_DETERMINISTIC,false}`` used to enable
    determinism for every run -- the opposite of the documented default.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def seed_everything(
    seed: int,
    rank: int,
    device: torch.device,
    platform: Any = None,
    deterministic: bool = False,
) -> None:
    """Seed every RNG a retrieval run draws from.

    ``rank`` is folded into the seed so replicas take different augmentation
    and dropout draws, while the *data* order stays reproducible because the
    datasets derive their shuffle seed from the same base seed + rank.

    ``deterministic`` additionally pins the algorithms that are otherwise
    free to pick a faster non-deterministic kernel.  It is off by default: on
    several vendor runtimes those kernels are the only implemented path, and
    turning them off raises at launch rather than at the first divergence.
    """

    seed += rank
    # Python's hash randomisation would otherwise vary set/dict iteration order
    # between processes; workers inherit this through the environment.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if platform is not None:
        platform.manual_seed_all(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def reproducibility_record(
    seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
    platform: Any = None,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Describe what was seeded, so a metrics.json can be matched to a rerun."""

    return {
        "seed": int(seed),
        "seed_per_rank": int(seed + rank),
        "world_size": int(world_size),
        "device": str(device),
        "platform": platform.name() if platform is not None else "cuda",
        "deterministic": bool(deterministic),
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def autocast_context(device: torch.device, dtype_name: str, platform: Any = None):
    if dtype_name == "fp32":
        return contextlib.nullcontext()
    dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    autocast_device = os.environ.get(
        "FLAGSCALE_AUTOCAST_DEVICE",
        platform.amp_device_type() if platform is not None else device.type,
    ).lower()
    if autocast_device == "auto":
        autocast_device = device.type
    try:
        return torch.autocast(device_type=autocast_device, dtype=dtype)
    except (RuntimeError, TypeError):
        # Some vendor runtimes expose AMP through their own torch backend and
        # do not register an autocast device string. In that case the model's
        # loaded dtype remains valid and the forward runs without an autocast
        # context instead of incorrectly forcing CUDA semantics.
        return contextlib.nullcontext()


def save_model(
    model: nn.Module,
    processor: Any,
    output_dir: Path,
    task: str,
    final: bool = True,
    source_dir: Path | None = None,
    sharded: bool = False,
) -> None:
    """Write a checkpoint.

    ``final`` marks the completed run: only then is ``training_complete.json``
    written, so a resume or an evaluator can tell a finished model from an
    intermediate epoch snapshot.

    Under FSDP2 the parameters are DTensors and ``state_dict()`` would yield
    shards.  The full state is gathered instead -- a collective, so every rank
    must call this function -- and written as a single ``model.safetensors``
    next to a copy of the source checkpoint's config/tokenizer files, which is
    the layout the evaluation entrypoint expects.  Materialising an unsharded
    model instead would double peak memory for an 8B adapter.
    """

    gathered = parallel_utils.full_state_dict(unwrap_model(model)) if sharded else None

    if not rank_zero():
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    if gathered is not None:
        import shutil

        from safetensors.torch import save_file

        if source_dir is None or not Path(source_dir).is_dir():
            raise RuntimeError(
                "FSDP2 checkpointing needs the source model directory to copy "
                "config/tokenizer files into the snapshot."
            )
        weights = {k.removeprefix("_orig_mod."): v for k, v in gathered.items()}
        save_file(weights, str(output_dir / "model.safetensors"))
        for item in Path(source_dir).iterdir():
            if item.is_file() and item.name not in {
                "model.safetensors",
                "pytorch_model.bin",
                "training_complete.json",
            }:
                shutil.copy2(item, output_dir / item.name)
        if processor is not None:
            with contextlib.suppress(Exception):
                processor.save_pretrained(output_dir)
    else:
        base = unwrap_model(model)
        if task == "clip":
            base.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
        else:
            base.save_pretrained(output_dir)

    if final:
        with (output_dir / "training_complete.json").open("w", encoding="utf-8") as file:
            json.dump({"task": task, "world_size": int(os.environ.get("WORLD_SIZE", "1"))}, file, indent=2)


def load_dotenv(path: Path | None = None) -> None:
    """Read KEY=VALUE pairs from the repo's .env into the environment.

    The file is gitignored and pushed separately (see sync.sh), so the training
    code is the only place that knows where the credentials live. Parsed here
    rather than with python-dotenv: one file, one format, no new dependency.

    Existing environment variables win, so a caller can still override.
    """

    candidate = path or (Path(__file__).resolve().parents[2] / ".env")
    if not candidate.is_file():
        return
    for line in candidate.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def init_wandb(train_cfg: DictConfig, model_cfg: DictConfig, output_dir: Path) -> Any:
    """Initialize W&B on rank zero when WANDB_API_KEY is present.

    W&B remains optional so offline/local experiments do not acquire a hard
    dependency. The API key is read by the W&B SDK from the environment and is
    never copied into the Hydra config or checkpoint directory.
    """

    load_dotenv()
    if not rank_zero() or os.environ.get("WANDB_DISABLED", "").lower() in {"1", "true", "yes"}:
        return None
    # The SDK may already be authenticated through ``wandb login``/netrc.
    # Require an explicit key only for the default mode; online mode is enough
    # to opt into a pre-authenticated run without putting a secret in a tmux
    # command line or repository file.
    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "").lower() != "online":
        log("WANDB_API_KEY is not set; continuing without W&B logging")
        return None
    try:
        import wandb

        run = wandb.init(
            project=str(train_cfg.get("wandb_project", "flagscale-retrieval")),
            name=str(train_cfg.get("wandb_name", train_cfg.get("name", "retrieval"))),
            dir=str(output_dir),
            config={
                "model_path": str(model_cfg.model_path),
                "task": str(train_cfg.task),
                "batch_size": int(train_cfg.batch_size),
                "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
                "epochs": int(train_cfg.get("epochs", 3)),
                "seed": int(train_cfg.get("seed", 42)),
            },
        )
        run.define_metric("train/optimizer_step")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="train/optimizer_step")
        run.define_metric("validation/*", step_metric="epoch")
        run.define_metric("test/*", step_metric="epoch")
        log(f"W&B run: {run.url}")
        return run
    except Exception as exc:  # pragma: no cover - depends on external W&B service
        log(f"W&B initialization failed ({exc}); continuing without remote logging")
        return None


def reduce_metrics(total_loss: float, count: int, device: torch.device) -> float:
    values = torch.tensor([total_loss, float(count)], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return float(values[0].item() / max(values[1].item(), 1.0))


def build_optimizer(
    optim_cfg: DictConfig,
    parameters: list[nn.Parameter],
    allow_adafactor: bool = True,
) -> torch.optim.Optimizer:
    """Create the configured optimizer for a model's trainable parameters.

    ``allow_adafactor=False`` substitutes AdamW when the config asks for
    Adafactor.  FSDP2 needs that: Adafactor's factored update does
    ``torch.norm(grad, dim=-2).square_()`` in place, and DTensor rejects an
    in-place op that would move a ``_NormPartial`` placement to ``Replicate``.
    Sharding is what freed the memory Adafactor was saving, so AdamW is both
    the working and the more accurate choice there.
    """

    name = str(optim_cfg.get("name", "adamw")).lower()
    lr = float(optim_cfg.lr)
    weight_decay = float(optim_cfg.get("weight_decay", 0.01))
    if name == "adafactor" and not allow_adafactor:
        log("optimizer 'adafactor' is incompatible with FSDP2 DTensor sharding; using AdamW")
        name = "adamw"
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adafactor":
        # Factored second-moment statistics keep 8B full fine-tuning within one
        # 80GB card: Adafactor state is O(rows+cols) per matrix instead of the
        # two full fp32 moments AdamW allocates.
        return torch.optim.Adafactor(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer {name!r}; use 'adamw' or 'adafactor'")


def build_lr_scheduler(
    optim_cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    total_steps: int | None = None,
) -> tuple[Any, int, dict[str, Any]]:
    """Create the configured LR schedule.

    A schedule is not optional. Without one the optimizer keeps ``lr`` for every
    step of every epoch; a 3-epoch 8B reranker run at a constant 1e-5 ended at
    train_loss 0.042 against validation_loss 0.377 (9x) and lost 1.5 points of
    R@5 to the base model. The earlier long run on the other platform fared
    worse: it peaked at eval loss 0.327 near step 6k and then diverged to 10-40
    with accuracy pinned at 0.5, again with a constant rate.

    Returns the scheduler, the number of warmup steps (0 when disabled) and a
    mutable state dict.  The retrieval datasets are streaming, so the number of
    optimizer steps in an epoch is only known after one has run; the caller
    updates ``state["total_steps"]`` at the first epoch boundary and the
    schedule re-anchors itself.  Passing ``total_steps`` explicitly (config
    ``train.lr_decay_steps``) skips that correction.
    """

    style = str(optim_cfg.get("lr_decay_style", "cosine")).lower()
    warmup_steps = int(optim_cfg.get("warmup_steps", 0) or 0)
    min_lr = float(optim_cfg.get("min_lr", 0.0) or 0.0)
    base_lr = float(optim_cfg.lr)
    # `total_steps=None` means the streaming dataset has not been measured yet.
    # Do NOT substitute a placeholder here: an earlier version did, which made
    # `known` true, disabled the re-anchor below, and decayed the rate to min_lr
    # by step 2 -- so every run trained at min_lr from start to finish. Hold at
    # ``base_lr`` until the length is measured instead.
    known_total = bool(total_steps)
    state: dict[str, Any] = {
        "total_steps": max(1, int(total_steps)) if known_total else 0,
        "known": known_total,
    }
    if known_total:
        # Only clamp against a total we can trust; clamping against a placeholder
        # silently reduced a 200-step warmup to 1 step.
        warmup_steps = min(max(0, warmup_steps), max(1, state["total_steps"] - 1))

    if min_lr > base_lr:
        raise ValueError(f"optimizer.min_lr ({min_lr}) must be <= optimizer.lr ({base_lr})")

    import math

    floor = (min_lr / base_lr) if base_lr else 0.0

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if style == "constant":
            return 1.0
        if not state["known"]:
            # Length still unknown: stay at base_lr rather than decaying against
            # a guess. The first epoch therefore runs at warmup->base_lr, and the
            # decay covers the remaining epochs.
            return 1.0
        total = max(1, int(state["total_steps"]))
        decay_steps = max(1, total - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        if style == "linear":
            return 1.0 - (1.0 - floor) * progress
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + (1.0 - floor) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    log(
        f"lr schedule: {style} {base_lr:g} -> {min_lr:g} "
        f"(total_steps={'configured' if state['known'] else 'estimated after epoch 1'}), "
        f"warmup {warmup_steps} steps"
    )
    return scheduler, warmup_steps, state


def train_epoch(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    epoch: int,
    task: str,
    device: torch.device,
    temperature: float,
    optimizer: torch.optim.Optimizer,
    trainable: list[nn.Parameter],
    grad_accum: int,
    max_grad_norm: float,
    dtype_name: str,
    log_every: int,
    wandb_run: Any = None,
    step_state: dict[str, int] | None = None,
    lr_scheduler: Any = None,
    warmup_steps: int = 0,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    # Reshuffle the streaming dataset each epoch. With DataLoader workers the
    # dataset is forked when the iterator starts, so this must run first.
    set_epoch = getattr(loader.dataset, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(epoch)
    micro_count = 0
    micro_batches = 0
    total_loss = 0.0
    window_loss = 0.0
    window_batches = 0
    optimizer_steps = 0
    # One-batch lookahead: the final micro-batch of an epoch must run with
    # gradient sync enabled even when it does not fill a whole accumulation
    # window, otherwise the partial-window step below would update from
    # gradients that were never reduced across ranks.
    stream = iter(loader)
    batch = next(stream, None)
    while batch is not None:
        nxt = next(stream, None)
        with autocast_context(device, dtype_name):
            raw_loss = loss_for_batch(model, processor, task, batch, device, temperature)
            loss = raw_loss / grad_accum
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(f"Non-finite training loss: {raw_loss.detach().item()}")
        # Only the last micro-step of an accumulation window needs the gradient
        # all-reduce; syncing on every micro-step multiplies collective traffic
        # by grad_accum, which dominates wall time once ranks span hosts.
        is_window_end = (micro_count + 1) >= grad_accum or nxt is None
        with contextlib.nullcontext() if is_window_end else parallel_utils.gradient_sync_disabled(model):
            loss.backward()
        batch = nxt
        micro_count += 1
        micro_batches += 1
        batch_loss = float(raw_loss.detach())
        total_loss += batch_loss
        window_loss += batch_loss
        window_batches += 1
        if micro_count < grad_accum:
            continue
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if lr_scheduler is not None:
            # The warmup lambdas already encode the ramp, so every step advances
            # the scheduler -- including the ones inside warmup.
            lr_scheduler.step()
        optimizer_steps += 1
        if step_state is not None:
            step_state["optimizer_step"] = step_state.get("optimizer_step", 0) + 1
        micro_count = 0
        if log_every > 0 and optimizer_steps % log_every == 0:
            global_step = (
                step_state.get("optimizer_step", optimizer_steps)
                if step_state is not None
                else optimizer_steps
            )
            # Report the average over the logging window rather than the whole
            # epoch.  A cumulative average hides the trend and makes the curve
            # look like it plateaus even while batches keep changing.
            window_average = window_loss / max(window_batches, 1)
            current_lr = optimizer.param_groups[0].get("lr")
            log(
                f"epoch={epoch} optimizer_step={optimizer_steps} "
                f"running_loss={total_loss / micro_batches:.6f} "
                f"window_loss={window_average:.6f} "
                # Printing lr makes a missing or frozen schedule visible; a
                # constant lr across a multi-epoch run is the failure this line
                # exists to catch.
                f"lr={float(current_lr):.3e}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step_loss": window_average,
                        "train/epoch_running_loss": total_loss / micro_batches,
                        "train/lr": float(optimizer.param_groups[0].get("lr", 0.0)),
                        "train/optimizer_step": global_step,
                        "epoch": epoch,
                    }
                )
            window_loss = 0.0
            window_batches = 0

    # Apply a final partial accumulation window instead of silently dropping
    # it.  Its gradient is already scaled by grad_accum, matching the normal
    # path and keeping epoch boundaries deterministic.
    if micro_count:
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1
        if step_state is not None:
            step_state["optimizer_step"] = step_state.get("optimizer_step", 0) + 1

    if optimizer_steps == 0:
        raise RuntimeError("Training dataset is empty for this split")
    return reduce_metrics(total_loss, micro_batches, device)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    task: str,
    device: torch.device,
    temperature: float,
    dtype_name: str,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for batch in loader:
        with autocast_context(device, dtype_name):
            loss = loss_for_batch(model, processor, task, batch, device, temperature)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite evaluation loss: {loss.detach().item()}")
        total_loss += float(loss.detach())
        count += 1
    if count == 0:
        raise RuntimeError("Evaluation dataset is empty for this split")
    return reduce_metrics(total_loss, count, device)


def train(config: DictConfig) -> None:
    train_cfg = config.train
    rank, world_size, device, platform = init_distributed(train_cfg.get("hardware", {}))
    task = str(train_cfg.task)
    seed = int(train_cfg.get("seed", 42))
    deterministic = _as_bool(train_cfg.get("deterministic", False))
    seed_everything(seed, rank, device, platform, deterministic=deterministic)
    seed_record = reproducibility_record(seed, rank, world_size, device, platform, deterministic)
    if rank_zero():
        log(
            "reproducibility: seed={seed} per_rank={seed_per_rank} deterministic={deterministic} "
            "world_size={world_size} platform={platform}".format(**seed_record)
        )
    model_cfg = train_cfg.model
    data_cfg = train_cfg.data
    optim_cfg = train_cfg.optimizer
    model_type = str(model_cfg.get("model_type", train_cfg.name))

    # Resolve parallelism and precision before the weights are loaded: loading
    # straight into the master dtype avoids holding a bf16 copy and an fp32 copy
    # of the model at the same time (which is what exhausted an 80 GB card when
    # an 8B adapter was converted in place).
    hardware_cfg = train_cfg.get("hardware", {})
    parallel_mode = str(train_cfg.get("parallel", hardware_cfg.get("parallel", "ddp"))).lower()
    master_dtype_name = str(train_cfg.get("master_dtype", "bf16")).lower()
    master_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
        master_dtype_name, torch.bfloat16
    )
    load_cfg = model_cfg
    if master_dtype_name != str(model_cfg.get("torch_dtype", "bf16")).lower():
        load_cfg = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        load_cfg.torch_dtype = master_dtype_name
        log(f"loading weights directly in master dtype {master_dtype_name}")
    model, processor = build_retrieval_model(model_type, load_cfg, device)
    if bool(model_cfg.get("gradient_checkpointing", False)):
        if unwrap_model(model).enable_gradient_checkpointing():
            log("gradient checkpointing enabled")
        else:
            log("gradient checkpointing requested but unsupported by this adapter")

    train_loader, validation_loader, test_loader = build_split_loaders(
        task,
        data_cfg,
        int(train_cfg.batch_size),
        pin_memory=bool(platform and platform.supports_pin_memory()),
        # Seed the train loader's sampler and workers from the same base seed so
        # a rerun with num_workers > 0 reproduces the batch order.
        seed=seed + rank,
    )

    dtype_name = str(model_cfg.get("torch_dtype", "bf16"))

    # ---- parallelism + precision -------------------------------------------
    # ``parallel`` selects how ranks are wired:
    #   ddp   - full replica per rank (the historical behaviour)
    #   fsdp2 - parameters/grads/optimizer state sharded across ranks
    # ``master_dtype`` decides what the optimizer sees.  With bf16 parameters
    # AdamW's moments are bf16 too, and a 1e-5 step falls below bf16's 2^-8
    # resolution, so the update is rounded away.  fp32 keeps the update; the
    # forward/backward still run in ``dtype_name`` via autocast (and via FSDP2's
    # MixedPrecisionPolicy when sharded).
    compute_dtype = {"fp16": torch.float16, "fp32": torch.float32}.get(dtype_name, torch.bfloat16)

    if parallel_mode == "fsdp2":
        if not parallel_utils.fsdp2_available():
            raise RuntimeError(
                "train.parallel=fsdp2 requires torch>=2.6 with composable FSDP; "
                "the installed torch does not expose torch.distributed._composable.fsdp."
            )
        model = parallel_utils.apply_fsdp2(
            model,
            device_type=device.type,
            world_size=max(world_size, 1),
            compute_dtype=compute_dtype,
            reduce_dtype=compute_dtype,
            log=log,
        )
    elif world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index]
            if platform is not None and platform.name() == "cuda"
            else None,
            # The adapters hold no buffers that need cross-rank synchronisation
            # (pooling is computed from the input mask), and the default
            # broadcast_buffers=True rewrites the tokenizer's input tensors in
            # place after every forward. Those are the same tensors the loss
            # reuses across the anchor/positive/negative passes, so the backward
            # of aten::embedding then fails with "modified by an inplace
            # operation ... expected version 5". Observed on the P800 vendor
            # torch; harmless to disable here either way.
            broadcast_buffers=False,
            # Poolers and the LM head are unused depending on the task/pooling
            # mode (bge-m3 uses CLS pooling and never touches `pooler`), which
            # DDP otherwise rejects when the reduction set changes shape.
            find_unused_parameters=True,
        )
    log(f"parallelism={parallel_mode} master_dtype={master_dtype_name} compute_dtype={dtype_name}")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters remain; set model.freeze_backbone=false")
    optimizer = build_optimizer(
        optim_cfg, trainable, allow_adafactor=(parallel_mode != "fsdp2")
    )
    epochs = int(train_cfg.get("epochs", 3))
    if epochs < 1:
        raise ValueError("train.epochs must be at least 1")
    lr_scheduler, warmup_steps, lr_sched_state = build_lr_scheduler(
        optim_cfg, optimizer, total_steps=optim_cfg.get("lr_decay_steps") or None
    )
    save_every_epoch = bool(train_cfg.get("save_every_epoch", True))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    temperature = float(train_cfg.get("temperature", 0.05))
    output_dir = Path(str(train_cfg.output_dir))
    wandb_run = init_wandb(train_cfg, model_cfg, output_dir)
    clip_grad = float(optim_cfg.get("max_grad_norm", 1.0))
    log_every = int(train_cfg.get("log_every", 10))
    wandb_step = {"optimizer_step": 0}
    training_started = time.time()
    last_train_loss = None
    last_validation_loss = None
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, processor, train_loader, epoch, task, device, temperature, optimizer,
            trainable, grad_accum, clip_grad, dtype_name, log_every,
            wandb_run=wandb_run, step_state=wandb_step,
            lr_scheduler=lr_scheduler, warmup_steps=warmup_steps,
        )
        # Re-anchor the schedule once the real epoch length is known. Without
        # this a streaming dataset would decay against a placeholder total, so
        # the cosine would either finish early or never reach min_lr.
        # Re-anchor at every epoch boundary: the streaming dataset's epoch
        # length is only measurable by running one, and re-measuring keeps a
        # short first epoch from fixing a wrong total for the whole run.
        steps_done = int(wandb_step["optimizer_step"])
        if steps_done > 0:
            refined = max(1, round(steps_done / epoch * epochs))
            if refined != lr_sched_state["total_steps"]:
                lr_sched_state["total_steps"] = refined
                lr_sched_state["known"] = True
                if rank_zero():
                    log(
                        f"lr schedule anchored at epoch {epoch}: {steps_done} steps so far "
                        f"-> total {refined} for {epochs} epochs"
                    )
        validation_loss = evaluate_epoch(
            model, processor, validation_loader, task, device, temperature, dtype_name
        )
        last_train_loss = train_loss
        last_validation_loss = validation_loss
        if rank_zero():
            log(
                f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} "
                f"validation_loss={validation_loss:.6f} device={device}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": train_loss,
                        "train/optimizer_step": wandb_step["optimizer_step"],
                        "validation/loss": validation_loss,
                        "epoch": epoch,
                    }
                )

        # Snapshot after every epoch. These runs take hours to days, so a crash
        # between epochs previously cost the entire run: the only save happened
        # after all epochs finished.
        if save_every_epoch:
            epoch_dir = output_dir / f"checkpoint-epoch-{epoch}"
            if dist.is_initialized():
                dist.barrier()
            save_model(
                model, processor, epoch_dir, task, final=False,
                source_dir=model_cfg.model_path, sharded=(parallel_mode == "fsdp2"),
            )
            log(f"saved epoch {epoch} checkpoint to {epoch_dir}")

    test_loss = evaluate_epoch(model, processor, test_loader, task, device, temperature, dtype_name)
    if rank_zero():
        log(f"test_loss={test_loss:.6f} epochs={epochs} elapsed={time.time() - training_started:.1f}s")
        if wandb_run is not None:
            wandb_run.log(
                {
                    "test/loss": test_loss,
                    "train/optimizer_step": wandb_step["optimizer_step"],
                    "epoch": epochs,
                }
            )

    if dist.is_initialized():
        dist.barrier()
    save_model(
        model, processor, output_dir, task,
        source_dir=model_cfg.model_path, sharded=(parallel_mode == "fsdp2"),
    )
    if rank_zero():
        with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "task": task,
                    "epochs": epochs,
                    "train_loss": last_train_loss,
                    "validation_loss": last_validation_loss,
                    "test_loss": test_loss,
                    "world_size": world_size,
                    # Everything needed to rerun this exact configuration. The
                    # seed alone is not enough: the same seed on a different
                    # torch version or worker count can still diverge.
                    "reproducibility": seed_record,
                    "config": {
                        "batch_size": int(train_cfg.batch_size),
                        "gradient_accumulation_steps": int(
                            train_cfg.get("gradient_accumulation_steps", 1)
                        ),
                        "parallel": parallel_mode,
                        "master_dtype": str(master_dtype).replace("torch.", ""),
                        "num_workers": int(data_cfg.get("num_workers", 0) or 0),
                        "shuffle_buffer": int(data_cfg.get("shuffle_buffer", 0) or 0),
                        "split_modulo": int(data_cfg.get("split_modulo", 20)),
                        "lr": float(optim_cfg.get("lr", 0.0)),
                    },
                },
                file,
                indent=2,
            )
    if wandb_run is not None:
        wandb_run.finish()
    log(f"saved checkpoint to {output_dir}")
    finish_distributed()


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config_file)
    # The native FlagScale backend writes the task configuration under train.
    if not config.get("train"):
        raise ValueError("Configuration must contain a train section")
    train(config)


if __name__ == "__main__":
    main()
