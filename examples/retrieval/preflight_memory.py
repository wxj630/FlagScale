"""Pre-flight a retrieval training config for single-card fit.

Loads the model exactly as the training entrypoint does (matching freeze flags,
gradient checkpointing and optimizer) and runs a few real optimizer steps,
reporting peak device memory.  Exit code 2 means the attempt ran out of device
memory, which lets a launch script step the batch size down until it fits.

Usage:
    python preflight_memory.py --config examples/retrieval/conf/train/qwen3_vl_embedding.yaml --batch-size 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time

import torch_fl  # noqa: F401  # must precede torch
import torch
from omegaconf import OmegaConf

from flagscale.models.retrieval import build_retrieval_model, unwrap_model
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch
from flagscale.train.train_retrieval import build_optimizer


def _gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().splitlines()
    return int(out[0]) if out else -1


class _Peak:
    def __init__(self) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, _gpu_used_mib())
            time.sleep(0.5)

    def __enter__(self):
        self._t.start(); return self

    def __exit__(self, *exc):
        self._stop.set(); self._t.join(timeout=2)
        self.peak = max(self.peak, _gpu_used_mib())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--device", default="flagos:0")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device(args.device)
    if str(device).startswith("flagos"):
        torch.flagos.set_device(0)

    batch_size = int(args.batch_size)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    model, processor = build_retrieval_model(cfg.model.model_type, cfg.model, device)
    if bool(cfg.model.get("gradient_checkpointing", False)):
        ok = unwrap_model(model).enable_gradient_checkpointing()
        print(f"gradient_checkpointing enabled={ok}", flush=True)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(
        f"trainable={trainable_count / 1e9:.3f}B of {total_count / 1e9:.3f}B",
        flush=True,
    )

    data_cfg = OmegaConf.create({
        "path": cfg.data.path,
        "train_split": cfg.data.get("train_split", "train"),
        "validation_split": cfg.data.get("validation_split", "validation"),
        "test_split": cfg.data.get("test_split", "test"),
        "num_negatives": int(cfg.data.get("num_negatives", 1)),
        "split_modulo": int(cfg.data.get("split_modulo", 20)),
        "shuffle_buffer": int(cfg.data.get("shuffle_buffer", 0)),
        "num_workers": int(cfg.data.get("num_workers", 0)),
        "max_samples": batch_size * grad_accum * args.steps + 8,
        "eval_max_samples": 8,
    })
    loader, _, _ = build_split_loaders(cfg.task, data_cfg, batch_size, pin_memory=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable)
    print(f"optimizer={type(optimizer).__name__}", flush=True)

    model.train()
    run_steps = 0
    micro = 0
    try:
        with _Peak() as peak:
            for batch in loader:
                with torch.autocast(device_type="flagos", dtype=torch.bfloat16):
                    loss = loss_for_batch(
                        model, processor, cfg.task, batch, device,
                        float(cfg.get("temperature", 0.05)),
                    )
                (loss / grad_accum).backward()
                micro += 1
                if micro < grad_accum:
                    continue
                micro = 0
                torch.nn.utils.clip_grad_norm_(
                    trainable, float(cfg.optimizer.get("max_grad_norm", 1.0))
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                run_steps += 1
                if run_steps >= args.steps:
                    break
            time.sleep(1.0)
    except torch.OutOfMemoryError:
        print(f"PREFLIGHT_OOM batch_size={batch_size}", flush=True)
        return 2

    print(
        f"PREFLIGHT_OK batch_size={batch_size} "
        f"gradient_accumulation_steps={grad_accum} steps={run_steps} peak_mib={peak.peak}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
