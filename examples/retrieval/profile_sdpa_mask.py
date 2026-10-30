"""Why does BGE-M3 attention cost 553 ms on PPU when sdpa is available?

The model builds XLMRobertaSdpaSelfAttention, and a bare SDPA call on the flagos
device is 2.4x faster than an explicit bmm+softmax.  Yet the real step profile
shows aten::bmm (347.6 ms) plus softmax (121.3 ms) and no fused attention
kernel.  The difference between the fast probe and the model is the padding
mask: XLMRobertaSdpaSelfAttention turns the 2D attention_mask into a 4D
additive mask, and the fast backends generally refuse that shape.

This isolates the mask's effect so the claim is measured, not inferred.
"""
import argparse, statistics, time
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=10)
ap.add_argument("--heads", type=int, default=16)
ap.add_argument("--seq", type=int, default=299)
ap.add_argument("--dim", type=int, default=64)
a = ap.parse_args()

dev = torch.device("cuda")
sync = torch.accelerator.synchronize if hasattr(torch, "accelerator") else torch.cuda.synchronize
B, H, S, D = a.batch, a.heads, a.seq, a.dim


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    sync()
    ts = []
    for _ in range(iters):
        sync(); t0 = time.perf_counter()
        fn()
        sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


q = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, device=dev, dtype=torch.bfloat16)

# The 4D additive mask XLMRobertaSdpaSelfAttention builds.
pad = torch.ones(B, S, device=dev, dtype=torch.bool)
pad[:, S // 2:] = False
mask4d = torch.zeros(B, 1, 1, S, device=dev, dtype=torch.bfloat16)
mask4d = mask4d.masked_fill(~pad[:, None, None, :], torch.finfo(torch.bfloat16).min)


def math_attn(mask=None):
    scale = 1.0 / (D ** 0.5)
    scores = torch.bmm((q * scale).reshape(B * H, S, D), k.reshape(B * H, S, D).transpose(1, 2))
    if mask is not None:
        scores = scores + mask.expand(B, H, 1, S).reshape(B * H, 1, S)
    probs = torch.softmax(scores, dim=-1)
    return torch.bmm(probs, v.reshape(B * H, S, D)).reshape(B, H, S, D)


print(f"shape B={B} H={H} S={S} D={D}")
cases = [
    ("sdpa, no mask", lambda: F.scaled_dot_product_attention(q, k, v)),
    ("sdpa, 4D additive mask", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask4d)),
    ("sdpa, bool 4D mask", lambda: F.scaled_dot_product_attention(
        q, k, v, attn_mask=(mask4d == 0).expand(B, H, S, S))),
    ("math bmm+softmax, no mask", lambda: math_attn()),
    ("math bmm+softmax, mask", lambda: math_attn(mask4d)),
]
for name, fn in cases:
    try:
        print(f"  {name:32}{bench(fn):8.2f} ms")
    except Exception as exc:
        print(f"  {name:32}FAILED {type(exc).__name__}: {str(exc)[:60]}")
