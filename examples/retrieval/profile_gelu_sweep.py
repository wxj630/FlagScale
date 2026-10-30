"""GELU cost vs row count, exact vs tanh.

In the real BGE-M3 step Kunlunxin spends 1097.9 ms in gelu/gelu_backward across
960 calls, while the same calls cost 42.5 ms on A100.  This sweep separates two
possible causes: a fixed per-call dispatch cost, or an arithmetic path whose cost
grows with the data.

Result (hidden 4096, fwd+bwd): A100 is start-up bound -- flat at ~0.12 ms from 128
through 1024 rows -- while Kunlunxin is purely linear at 0.64 us/row.  So the
vendor's `gelu_highprecision_nv` path is instruction-bound, not overhead-bound.
approximate="tanh" is 1.65x faster on Kunlunxin (1.951 -> 1.162 ms at 2990 rows)
and makes no difference on A100 (0.217 -> 0.203 ms).
"""
import argparse, statistics, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--hidden", type=int, default=4096)
ap.add_argument("--rows", type=int, nargs="+", default=[128, 256, 512, 1024, 2990, 5980])
a = ap.parse_args()

dev = torch.device("cuda")
H = a.hidden


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        x = torch.randn(1, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
        fn(x).sum().backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        x = torch.randn(1, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(x).sum().backward()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def timed_rows(fn, rows, warmup=5, iters=20):
    x = torch.randn(rows, H, device=dev, dtype=torch.bfloat16, requires_grad=True)
    for _ in range(warmup):
        fn(x).sum().backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn(x).sum().backward()
        torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


print(f"hidden={H}  fwd+bwd per call, median of 20")
print(f"{'rows':>8}{'exact ms':>11}{'tanh ms':>11}{'exact us/row':>14}{'tanh us/row':>13}")
for r in a.rows:
    e = timed_rows(lambda x: F.gelu(x), r)
    t = timed_rows(lambda x: F.gelu(x, approximate="tanh"), r)
    print(f"{r:8d}{e:11.3f}{t:11.3f}{e*1000/r:14.2f}{t*1000/r:13.2f}", flush=True)
