#!/usr/bin/env python3
"""Benchmark square GEMM on Ascend NPU: fp32 / fp16 / bf16.

Run with one idle chip visible, e.g.:
    ASCEND_RT_VISIBLE_DEVICES=8 python scripts/bench_gemm_npu.py
"""

import json
import platform
import statistics
import time
from datetime import datetime

import torch
import torch_npu  # noqa: F401  (registers the NPU backend)

DTYPES = [
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
]
SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]


def bench_once(a, b):
    """Return elapsed milliseconds for one matmul, using NPU events."""
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    torch.matmul(a, b)
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end)


def bench(dtype, m, n, k, warmup=3, iters=None):
    if iters is None:
        iters = 10 if m >= 8192 else 20
    a = torch.randn(m, k, device="npu", dtype=dtype)
    b = torch.randn(k, n, device="npu", dtype=dtype)
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.npu.synchronize()

    samples = [bench_once(a, b) for _ in range(iters)]
    samples.sort()
    median_ms = statistics.median(samples)
    flops = 2.0 * m * n * k
    tflops = flops / (median_ms * 1e-3) / 1e12

    out = {
        "dtype": str(dtype).replace("torch.", ""),
        "m": m,
        "n": n,
        "k": k,
        "median_ms": round(median_ms, 4),
        "min_ms": round(samples[0], 4),
        "max_ms": round(samples[-1], 4),
        "tflops": round(tflops, 3),
        "iters": iters,
    }
    del a, b
    torch.npu.empty_cache()
    return out


def correctness():
    """Small check that NPU matmul matches CPU fp32 within dtype tolerance."""
    torch.manual_seed(0)
    a = torch.randn(256, 256)
    b = torch.randn(256, 256)
    ref = a @ b
    checks = {}
    for name, dtype in DTYPES:
        got = torch.matmul(a.npu().to(dtype), b.npu().to(dtype)).float().cpu()
        rel = ((got - ref).norm() / ref.norm()).item()
        checks[name] = {
            "relative_frobenius_error": rel,
            "abs_max": (got - ref).abs().max().item(),
        }
    return checks


def main():
    if not torch.npu.is_available():
        raise SystemExit("NPU is not available: check CANN env / torch_npu")
    torch.npu.set_device(0)
    device_name = torch.npu.get_device_name(0)
    props = torch.npu.get_device_properties(0)

    print(f"time          : {datetime.now().isoformat(timespec='seconds')}")
    print(f"host          : {platform.node()}")
    print(f"torch         : {torch.__version__}")
    print(f"torch_npu     : {torch_npu.__version__}")
    print(f"device        : {device_name}")
    print(f"total_memory  : {props.total_memory} MB")
    print(f"cube/vector   : {props.cube_core_num} / {props.vector_core_num}")
    print(f"visible_chip  : 0 (physical selection via ASCEND_RT_VISIBLE_DEVICES)")
    print()

    print("correctness (256x256, vs CPU fp32):")
    checks = correctness()
    for name, check in checks.items():
        print(
            f"  {name:4s} rel_err={check['relative_frobenius_error']:.3e} "
            f"abs_max={check['abs_max']:.3e}"
        )
    print()

    results = []
    header = f"{'dtype':5s} {'M':>5s} {'N':>5s} {'K':>5s} {'median_ms':>10s} {'TFLOPS':>8s} {'min_ms':>8s} {'max_ms':>8s}"
    print(header)
    print("-" * len(header))
    for name, dtype in DTYPES:
        for m, n, k in SHAPES:
            r = bench(dtype, m, n, k)
            results.append(r)
            print(
                f"{name:5s} {m:5d} {n:5d} {k:5d} {r['median_ms']:10.3f} "
                f"{r['tflops']:8.2f} {r['min_ms']:8.3f} {r['max_ms']:8.3f}",
                flush=True,
            )

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "device": device_name,
        "total_memory_mb": props.total_memory,
        "cube_core_num": props.cube_core_num,
        "vector_core_num": props.vector_core_num,
        "correctness": checks,
        "results": results,
    }
    out_path = f"logs/gemm-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON written to {out_path}")


if __name__ == "__main__":
    main()
