"""Break one real training step into segments, on the real config.

The docs once described the wall-clock-minus-kernel residual as "the data
pipeline", but that subtraction also contains CPU tokenisation, the optimizer
update, gradient clipping and every host-side stall, and it was never measured.
This script measures each part directly, and takes the kernel total from a
profiler run over a whole optimizer step instead of extrapolating from a small
synthetic batch.

    python examples/retrieval/profile_step_segments.py \
        --conf examples/retrieval/conf/train/bge_m3.yaml \
        --steps 3 --skip 2 --device cuda

Pass --device flagos on Torch-FL, cuda on native CUDA and on the Kunlunxin shim.
Measured 2026-10-30 (BGE-M3, batch 10 x accum 4 = 200 texts/step):

    A100      1830 ms/step, 85.6% device compute, 6.0% tokenize, 0.04% dataloader
    Kunlunxin 3368 ms/step, 89.4% device compute, 3.2% tokenize, 0.03% dataloader
"""
import argparse, json, os, time, statistics
import torch
from omegaconf import OmegaConf


def sync():
    """Synchronise whatever accelerator this build exposes.

    torch.cuda works for native CUDA and for the Kunlunxin shim; Torch-FL's
    PrivateUse1 device needs its own module, and newer torch has a generic
    entrypoint that covers both.
    """
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", required=True)
    ap.add_argument("--steps", type=int, default=3, help="optimizer steps to time")
    ap.add_argument("--skip", type=int, default=2, help="warm-up steps")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto", help="auto | cuda | flagos | npu")
    a = ap.parse_args()

    from flagscale.train.datasets.retrieval import build_split_loaders
    from flagscale.train.losses.retrieval import loss_for_batch, move_features
    from flagscale.train.train_retrieval import build_optimizer, autocast_context
    from flagscale.models.retrieval import build_retrieval_model

    cfg = OmegaConf.load(a.conf)
    task = str(cfg.task)
    data_cfg = cfg.data
    if a.workers is not None:
        OmegaConf.set_struct(data_cfg, False)
        data_cfg.num_workers = a.workers
    batch_rows = int(cfg.batch_size)
    accum = int(cfg.gradient_accumulation_steps)
    texts = batch_rows * accum * 5           # 1 anchor + 1 pos + 3 neg per row
    from flagscale.train.train_retrieval import init_distributed
    os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
    _rank, _world, dev, platform = init_distributed(
        OmegaConf.create({"device_type": a.device})
    )
    dtype_name = str(cfg.model.get("torch_dtype", "bf16"))

    model, processor = build_retrieval_model(str(cfg.model.model_type), cfg.model, dev)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = build_optimizer(cfg.optimizer, trainable, allow_adafactor=False)
    loaders = build_split_loaders(task, data_cfg, batch_rows, pin_memory=True, seed=42)
    stream = iter(loaders[0])
    print(f"device={dev} platform={platform.name() if platform else dev.type}", flush=True)

    seg = {k: [] for k in ("data", "tokenize", "h2d", "forward", "backward", "clip_step")}
    prof = None
    if not a.no_profile:
        from torch.profiler import profile, ProfilerActivity
        prof = profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU])

    total_wall = []
    for step in range(a.skip + a.steps):
        recording = prof is not None and step == a.skip + a.steps - 1
        t_step = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        if recording:
            prof.start()
        for _ in range(accum):
            t0 = time.perf_counter(); batch = next(stream); seg["data"].append(time.perf_counter() - t0)

            # Tokenisation is host-only, so timing it with the device idle gives
            # its true cost.  The real step then runs loss_for_batch, which
            # tokenises again internally; ``forward`` is that whole call, and the
            # two numbers are subtracted to isolate compute.
            sync()
            t0 = time.perf_counter()
            feats = {k: model.prepare_embedding_inputs(batch[k]) for k in sorted(batch)}
            seg["tokenize"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            {k: move_features(v, dev) for k, v in feats.items()}
            sync()
            seg["h2d"].append(time.perf_counter() - t0)
            del feats

            sync(); t0 = time.perf_counter()
            with autocast_context(dev, dtype_name):
                loss = loss_for_batch(model, processor, task, batch, dev, float(cfg.temperature))
            sync(); seg["forward"].append(time.perf_counter() - t0)

            sync(); t0 = time.perf_counter()
            (loss / accum).backward()
            sync(); seg["backward"].append(time.perf_counter() - t0)

        sync(); t0 = time.perf_counter()
        if float(cfg.optimizer.get("max_grad_norm", 0) or 0) > 0:
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.optimizer.max_grad_norm))
        optimizer.step()
        sync(); seg["clip_step"].append(time.perf_counter() - t0)
        if recording:
            prof.stop()
        if step >= a.skip:
            total_wall.append(time.perf_counter() - t_step)
            print(f"  step {step}: {total_wall[-1]*1000:.0f} ms", flush=True)

    med = {k: statistics.median(v) * 1000 for k, v in seg.items()}
    # ``forward`` already contains one tokenise + one H2D, so only the three
    # per-micro segments plus the optimizer belong to the wall time.
    per_step = {k: med[k] * accum for k in ("data", "forward", "backward")}
    per_step["clip_step"] = med["clip_step"]
    per_step["tokenize_seen_inside_forward"] = med["tokenize"] * accum
    per_step["h2d_seen_inside_forward"] = med["h2d"] * accum
    per_step["forward_pure_compute"] = (med["forward"] - med["tokenize"] - med["h2d"]) * accum
    wall_ms = statistics.median(total_wall) * 1000
    acc = sum(per_step[k] for k in ("data", "forward", "backward", "clip_step"))

    out = {"batch_rows": batch_rows, "accum": accum, "texts_per_step": texts,
           "wall_ms": wall_ms, "segments_ms_median": med,
           "per_step_ms": per_step, "sum_segments_ms": acc,
           "unaccounted_ms": wall_ms - acc}

    if prof is not None:
        ops = {}
        for e in prof.key_averages():
            if e.self_device_time_total <= 0:
                continue
            ops[e.key] = {"calls": e.count, "ms": e.self_device_time_total / 1000.0}
        # aten:: wrappers and the kernels they launch both report device time, so
        # adding every entry double counts.  Prefer the aten level when the build
        # emits it; vendor builds report bare kernel names only.
        aten = {k: v for k, v in ops.items() if k.startswith("aten::")}
        level = aten if aten else ops
        kernel_ms = sum(v["ms"] for v in level.values())
        out["profiled_steps"] = 1
        out["kernel_ms"] = kernel_ms
        out["kernel_level"] = "aten" if aten else "kernel"
        out["kernel_ms_all_entries"] = sum(v["ms"] for v in ops.values())
        out["ops"] = ops
        out["kernel_pct"] = kernel_ms / wall_ms * 100

    print(f"\n=== batch {batch_rows} rows x accum {accum} = {texts} texts/step ===")
    print(f"wall/step          {wall_ms:9.1f} ms")
    print(f"{'segment':34}{'per step':>11}{'% wall':>9}")
    rows = [
        ("data (next(stream))", per_step["data"]),
        ("forward (incl tokenize+h2d)", per_step["forward"]),
        ("  . tokenize alone", per_step["tokenize_seen_inside_forward"]),
        ("  . H2D alone", per_step["h2d_seen_inside_forward"]),
        ("  . residual = device compute", per_step["forward_pure_compute"]),
        ("backward", per_step["backward"]),
        ("clip + optimizer.step", per_step["clip_step"]),
    ]
    for label, ms in rows:
        print(f"{label:34}{ms:11.2f}{ms/wall_ms*100:9.1f}")
    print(f"{'sum':34}{acc:11.2f}{acc/wall_ms*100:9.1f}")
    print(f"{'unaccounted':34}{wall_ms-acc:11.2f}{(wall_ms-acc)/wall_ms*100:9.1f}")
    device_compute = per_step["forward_pure_compute"] + per_step["backward"]
    print(f"{'device compute (fwd+bwd)':34}{device_compute:11.2f}{device_compute/wall_ms*100:9.1f}")
    if prof is not None:
        print(f"kernel ({out['kernel_level']})  {out['kernel_ms']:9.1f} {out['kernel_pct']:8.1f}% of wall"
              f"   [all entries {out['kernel_ms_all_entries']:.1f}]")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
