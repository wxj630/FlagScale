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


def seed_everything(seed: int, rank: int, device: torch.device, platform: Any = None) -> None:
    seed += rank
    random.seed(seed)
    torch.manual_seed(seed)
    if platform is not None:
        platform.manual_seed_all(seed)


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


def save_model(model: nn.Module, processor: Any, output_dir: Path, task: str) -> None:
    if not rank_zero():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    base = unwrap_model(model)
    if task == "clip":
        base.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
    else:
        base.save_pretrained(output_dir)
    with (output_dir / "training_complete.json").open("w", encoding="utf-8") as file:
        json.dump({"task": task, "world_size": int(os.environ.get("WORLD_SIZE", "1"))}, file, indent=2)


def init_wandb(train_cfg: DictConfig, model_cfg: DictConfig, output_dir: Path) -> Any:
    """Initialize W&B on rank zero when WANDB_API_KEY is present.

    W&B remains optional so offline/local experiments do not acquire a hard
    dependency. The API key is read by the W&B SDK from the environment and is
    never copied into the Hydra config or checkpoint directory.
    """

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
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_count = 0
    micro_batches = 0
    total_loss = 0.0
    window_loss = 0.0
    window_batches = 0
    optimizer_steps = 0
    for batch in loader:
        with autocast_context(device, dtype_name):
            raw_loss = loss_for_batch(model, processor, task, batch, device, temperature)
            loss = raw_loss / grad_accum
        if not torch.isfinite(raw_loss):
            raise FloatingPointError(f"Non-finite training loss: {raw_loss.detach().item()}")
        loss.backward()
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
            log(
                f"epoch={epoch} optimizer_step={optimizer_steps} "
                f"running_loss={total_loss / micro_batches:.6f} "
                f"window_loss={window_average:.6f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/step_loss": window_average,
                        "train/epoch_running_loss": total_loss / micro_batches,
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
    seed_everything(int(train_cfg.get("seed", 42)), rank, device, platform)
    model_cfg = train_cfg.model
    data_cfg = train_cfg.data
    optim_cfg = train_cfg.optimizer
    model_type = str(model_cfg.get("model_type", train_cfg.name))
    model, processor = build_retrieval_model(model_type, model_cfg, device)

    train_loader, validation_loader, test_loader = build_split_loaders(
        task,
        data_cfg,
        int(train_cfg.batch_size),
        pin_memory=bool(platform and platform.supports_pin_memory()),
    )

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[device.index]
            if platform is not None and platform.name() == "cuda"
            else None,
            find_unused_parameters=False,
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters remain; set model.freeze_backbone=false")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(optim_cfg.lr),
        weight_decay=float(optim_cfg.get("weight_decay", 0.01)),
    )
    epochs = int(train_cfg.get("epochs", 3))
    if epochs < 1:
        raise ValueError("train.epochs must be at least 1")
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    temperature = float(train_cfg.get("temperature", 0.05))
    dtype_name = str(model_cfg.get("torch_dtype", "bf16"))
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
    save_model(model, processor, output_dir, task)
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
