"""Measure throughput per step on the real training config, and its spread.

The per-chip docs report "fastest / average / slowest" throughput.  On Kunlunxin
and PPU that came from wandb step timestamps, but the A100 runs predate a
working WANDB_API_KEY and their logs carry no per-step timestamps at all, so the
baseline had only an end-to-end number and no spread to compare against.

This produces the same statistic the wandb-derived numbers use: group steps into
fixed windows, convert each window to samples/s, drop the first and last 5% of
windows (start-up and tear-down), then report min / median / max.

    python examples/retrieval/measure_throughput.py \
        --conf examples/retrieval/conf/train/bge_m3.yaml \
        --steps 200 --skip 20 --device cuda

Samples per step follow the same convention as the docs: embedding counts every
text in the batch (batch x accum x 5: anchor, positive, 3 negatives), clip counts
pairs, reranker counts pairs across all ranks.
"""
import argparse, json, os, statistics, time
import torch
from omegaconf import OmegaConf


def sync():
    accel = getattr(torch, "accelerator", None)
    if accel is not None:
        try:
            accel.synchronize()
            return
        except Exception:
            pass
    for mod in ("cuda", "flagos", "npu", "musa"):
        m = getattr(torch, mod, None)
        if m is not None and hasattr(m, "synchronize"):
            try:
                m.synchronize()
                return
            except Exception:
                continue


def samples_per_step(cfg, task, world_size):
    rows = int(cfg.batch_size) * int(cfg.gradient_accumulation_steps)
    if task == "embedding":
        negatives = int(cfg.data.get("num_negatives", 3) or 3)
        return rows * (2 + negatives)
    if task in {"clip", "vl_embedding"}:
        return rows * world_size
    if task == "reranker":
        return rows * world_size
    raise ValueError(f"unknown task {task!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--skip", type=int, default=20, help="warm-up steps")
    ap.add_argument("--window", type=int, default=10, help="steps per throughput window")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from flagscale.train.datasets.retrieval import build_split_loaders
    from flagscale.train.losses.retrieval import loss_for_batch
    from flagscale.train.train_retrieval import build_optimizer, autocast_context, init_distributed
    from flagscale.models.retrieval import build_retrieval_model

    os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
    rank, world, dev, platform = init_distributed(OmegaConf.create({"device_type": a.device}))

    cfg = OmegaConf.load(a.conf)
    task = str(cfg.task)
    data_cfg = cfg.data
    if a.workers is not None:
        OmegaConf.set_struct(data_cfg, False)
        data_cfg.num_workers = a.workers
    dtype_name = str(cfg.model.get("torch_dtype", "bf16"))
    accum = int(cfg.gradient_accumulation_steps)
    per_step = samples_per_step(cfg, task, world)

    model, processor = build_retrieval_model(str(cfg.model.model_type), cfg.model, dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable, allow_adafactor=False)
    loaders = build_split_loaders(task, data_cfg, int(cfg.batch_size), pin_memory=True, seed=42)
    stream = iter(loaders[0])

    if rank == 0:
        print(f"device={dev} platform={platform.name() if platform else dev.type} "
              f"world={world} samples/step={per_step}", flush=True)

    step_times = []
    for step in range(a.skip + a.steps):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            batch = next(stream)
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, task, batch, dev, float(cfg.temperature))
            (loss / accum).backward()
        if float(cfg.optimizer.get("max_grad_norm", 0) or 0) > 0:
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optimizer.max_grad_norm))
        optimizer.step()
        sync()
        if step >= a.skip:
            step_times.append(time.perf_counter() - t0)

    # Window throughputs, then trim 5% at both ends like the wandb-derived numbers.
    windows = []
    for i in range(0, len(step_times) - a.window + 1, a.window):
        chunk = step_times[i:i + a.window]
        windows.append(per_step * len(chunk) / sum(chunk))
    trim = int(len(windows) * 0.05)
    kept = windows[trim:len(windows) - trim] if len(windows) > 2 * trim + 2 else windows

    out = {
        "conf": a.conf, "task": task, "world_size": world,
        "samples_per_step": per_step,
        "steps_timed": len(step_times),
        "windows": len(windows),
        "windows_kept": len(kept),
        "step_ms_median": statistics.median(step_times) * 1000,
        "slowest": min(kept),          # slowest window = lowest throughput
        "average": statistics.median(kept),
        "fastest": max(kept),
        "end_to_end": per_step * len(step_times) / sum(step_times),
    }
    if rank == 0:
        print(f"steps={out['steps_timed']} windows={out['windows']} kept={out['windows_kept']}")
        print(f"step median {out['step_ms_median']:.0f} ms")
        print(f"  slowest {out['slowest']:8.1f} samples/s")
        print(f"  average {out['average']:8.1f} samples/s")
        print(f"  fastest {out['fastest']:8.1f} samples/s")
        print(f"  (whole-run equivalent {out['end_to_end']:.1f} samples/s)")
    if a.out and rank == 0:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
