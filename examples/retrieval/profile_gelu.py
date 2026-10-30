"""Why is GELU 25.8x slower on Kunlunxin, and can a different formulation avoid it?

In the real BGE-M3 step, `aten::gelu` and `aten::gelu_backward` cost 42.5 ms on
A100 and 1097.9 ms on Kunlunxin for the identical 480+480 calls.  The vendor
dispatcher picks `gelu_highprecision_nv`, a path whose cost grows linearly with
row count rather than being launch-bound.

aten::gelu is on autocast's fp32 list, so a bf16 model still computes GELU in
fp32.  This measures the formulations that could avoid that kernel at the exact
tile size the real step uses.
"""
import argparse, statistics, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--rows", type=int, default=2990, help="padded rows in one micro-batch group")
ap.add_argument("--hidden", type=int, default=4096)
a = ap.parse_args()

dev = torch.device("cuda")
B, H = a.rows, a.hidden
n_bytes = B * H * 2


def bench_fwd(fn, warmup=5, iters=20):
    x = torch.randn(B, H, device=dev, dtype=torch.bfloat16)
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def bench_bwd(fn, warmup=5, iters=20):
    for _ in range(warmup):
        x = torch.randn(B, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
        fn(x).sum().backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        x = torch.randn(B, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(x).sum().backward()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


bf = torch.randn(B, H, device=dev, dtype=torch.bfloat16)
fp = bf.float()
C = 0.7978845608028654   # sqrt(2/pi)


def gelu_tanh_manual(x, dt):
    xf = x.to(dt)
    inner = C * (xf + 0.044715 * xf * xf * xf)
    return (0.5 * xf * (1.0 + torch.tanh(inner))).to(x.dtype)


cases = {
    "gelu bf16 in, exact": lambda x: F.gelu(x),
    "gelu fp32 in, exact": lambda x: F.gelu(x.float()),
    "gelu bf16 in, tanh": lambda x: F.gelu(x, approximate="tanh"),
    "gelu fp32 in, tanh": lambda x: F.gelu(x.float(), approximate="tanh"),
    "manual tanh chain bf16": lambda x: gelu_tanh_manual(x, torch.bfloat16),
    "manual tanh chain fp32": lambda x: gelu_tanh_manual(x, torch.float32),
}

print(f"shape [{B}, {H}] bf16  ({n_bytes/2**20:.0f} MiB in)")
for name, fn in cases.items():
    try:
        ms = bench_fwd(fn)
        gbs = n_bytes * 2 / (ms / 1000) / 1e9
        print(f"  fwd  {name:26} {ms:8.2f} ms  {gbs:7.1f} GB/s")
    except Exception as exc:
        print(f"  fwd  {name:26} SKIP {type(exc).__name__}: {str(exc)[:40]}")

# What the model actually calls: check whether autocast promotes the input.
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    try:
        out = F.gelu(bf)
        print(f"  autocast: bf16 in -> {out.dtype} out")
    except Exception as exc:
        print(f"  autocast probe failed: {exc}")

print()
for name, fn in (("exact", lambda x: F.gelu(x)),
                 ("tanh", lambda x: F.gelu(x, approximate="tanh")),
                 ("manual tanh bf16", lambda x: gelu_tanh_manual(x, torch.bfloat16))):
    try:
        print(f"  fwd+bwd {name:20} {bench_bwd(fn):8.2f} ms")
    except Exception as exc:
        print(f"  fwd+bwd {name:20} SKIP {str(exc)[:40]}")
