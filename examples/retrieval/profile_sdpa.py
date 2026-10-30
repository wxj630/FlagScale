"""Does PPU have a fused attention path, and what does the model actually use?

The real BGE-M3 step shows PPU spending 347.6 ms in aten::bmm plus 121.3 ms in
softmax where A100 spends 319.6 ms in _efficient_attention_* (one fused kernel
pair).  That pattern means attention fell back to the math path.

This checks whether scaled_dot_product_attention is implemented on this build
and how much slower the math path is at the shapes the step uses.
"""
import argparse, statistics, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--seq", type=int, default=299)
ap.add_argument("--dim", type=int, default=64)
ap.add_argument("--batch", type=int, default=10)
a = ap.parse_args()

dev = torch.device("cuda")
s = torch.accelerator.synchronize if hasattr(torch, "accelerator") else torch.cuda.synchronize


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    s()
    ts = []
    for _ in range(iters):
        s(); t0 = time.perf_counter()
        fn()
        s(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


B, H, S, D = a.batch, a.heads, a.seq, a.dim
q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)


def math_attn():
    scale = 1.0 / (D ** 0.5)
    scores = torch.bmm(
        (q * scale).reshape(B * H, S, D),
        k.reshape(B * H, S, D).transpose(1, 2),
    )
    probs = torch.softmax(scores, dim=-1)
    return torch.bmm(probs, v.reshape(B * H, S, D)).reshape(B, H, S, D)


print(f"shape B={B} H={H} S={S} D={D}")
print(f"  math path (bmm+softmax)  {bench(math_attn):8.2f} ms")

for name, fn in (
    ("sdpa default", lambda: F.scaled_dot_product_attention(q, k, v)),
    ("sdpa math backend", lambda: F.scaled_dot_product_attention(
        q, k, v, scale=None) ),
):
    try:
        print(f"  {name:24}{bench(fn):8.2f} ms")
        break
    except Exception as exc:
        print(f"  {name:24} FAILED {type(exc).__name__}: {str(exc)[:70]}")

# Which backends does this build claim?
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    avail = []
    for b in SDPBackend:
        try:
            with sdpa_kernel(b):
                F.scaled_dot_product_attention(q, k, v)
            avail.append(b.name)
        except Exception:
            pass
    print(f"  available sdpa backends: {avail or 'none'}")
except Exception as exc:
    print(f"  sdpa backend probe unavailable: {type(exc).__name__}: {str(exc)[:60]}")
