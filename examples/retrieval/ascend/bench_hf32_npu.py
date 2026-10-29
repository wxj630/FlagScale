#!/usr/bin/env python3
"""FP32 GEMM on Ascend with HF32 disabled/enabled (Huawei's TF32 analogue)."""

import json
import platform
import statistics
from datetime import datetime

import torch
import torch_npu  # noqa: F401

SHAPES = [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]


def bench(m, n, k, hf32, warmup=5, iters=None):
    if iters is None:
        iters = 10 if m >= 8192 else 20
    torch.npu.matmul.allow_hf32 = hf32
    a = torch.randn(m, k, device="npu", dtype=torch.float32)
    b = torch.randn(k, n, device="npu", dtype=torch.float32)
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.npu.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        torch.matmul(a, b)
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    med = statistics.median(samples)
    tflops = 2.0 * m * n * k / (med * 1e-3) / 1e12
    del a, b
    torch.npu.empty_cache()
    return {
        "shape": [m, n, k],
        "hf32": hf32,
        "median_ms": round(med, 4),
        "min_ms": round(samples[0], 4),
        "max_ms": round(samples[-1], 4),
        "tflops": round(tflops, 3),
        "iters": iters,
    }


def correctness(hf32):
    torch.manual_seed(0)
    a = torch.randn(512, 512)
    b = torch.randn(512, 512)
    ref = a @ b
    torch.npu.matmul.allow_hf32 = hf32
    got = (a.npu() @ b.npu()).cpu()
    return {
        "hf32": hf32,
        "relative_frobenius_error": ((got - ref).norm() / ref.norm()).item(),
        "abs_max": (got - ref).abs().max().item(),
    }


def main():
    torch.npu.set_device(0)
    print(f"time       : {datetime.now().isoformat(timespec='seconds')}")
    print(f"host       : {platform.node()}")
    print(f"device     : {torch.npu.get_device_name(0)}")
    print(f"torch      : {torch.__version__}  torch_npu: {torch_npu.__version__}")
    print(f"default    : allow_hf32={torch.npu.matmul.allow_hf32}")
    print()

    checks = []
    print("correctness (512x512 vs CPU fp32):")
    for hf32 in (False, True):
        c = correctness(hf32)
        checks.append(c)
        print(f"  hf32={str(hf32):5s} rel_err={c['relative_frobenius_error']:.3e} abs_max={c['abs_max']:.3e}")
    print()

    results = []
    header = f"{'hf32':5s} {'M':>5s} {'N':>5s} {'K':>5s} {'median_ms':>10s} {'TFLOPS':>8s}"
    print(header)
    print("-" * len(header))
    for hf32 in (False, True):
        for m, n, k in SHAPES:
            r = bench(m, n, k, hf32)
            results.append(r)
            print(
                f"{str(hf32):5s} {m:5d} {n:5d} {k:5d} {r['median_ms']:10.3f} {r['tflops']:8.2f}",
                flush=True,
            )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "device": torch.npu.get_device_name(0),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "correctness": checks,
        "results": results,
    }
    path = f"logs/hf32-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON written to {path}")


if __name__ == "__main__":
    main()
