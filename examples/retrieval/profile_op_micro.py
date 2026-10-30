"""Time the dominant training ops one by one, identically on every chip.

Profiler tracing is unavailable on the vendor stacks (Kunlunxin's CUPTI needs an
env var whose spelling differs per build), so the comparison is done by timing
each op in a loop instead.  Every op here was picked from the A100 profile of
the three retrieval models, ordered by its share of device time.

Reported number: milliseconds for one call, median of N iterations, after a
warm-up that lets the vendor runtime finish its per-shape JIT.
"""
import argparse, json, statistics, time
import torch
import torch.nn.functional as F


def bench(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    dt = torch.bfloat16
    B, S, H, I, V = 8, 512, 1024, 4096, 151669
    x = torch.randn(B, S, H, device=dev, dtype=dt)
    w_h = torch.randn(H, H, device=dev, dtype=dt)
    w_i = torch.randn(H, I * 2, device=dev, dtype=dt)
    w_v = torch.randn(H, V, device=dev, dtype=dt)
    w_n = torch.randn(H, device=dev, dtype=dt)
    q = torch.randn(B * S, 32, 128, device=dev, dtype=dt)
    k = torch.randn(B * S, 8, 128, device=dev, dtype=dt)

    ops = {}
    # 1. GEMM: the dominant op in all three models
    ops["mm_hidden"] = lambda: torch.mm(x.view(-1, H), w_h)
    ops["mm_ffn"] = lambda: torch.mm(x.view(-1, H), w_i)
    # lm_head: [tokens, hidden] x [hidden, vocab] -- the biggest single GEMM
    ops["mm_lm_head"] = lambda: torch.mm(x.view(-1, H), w_v)
    # 2. attention core
    ops["flash_attn_fwd"] = lambda: F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0), enable_gqa=True)
    # 3. normalisation and activation: these are what FlagGems fuses
    ops["rms_norm"] = lambda: F.rms_norm(x, (H,), w_n, 1e-6)
    ops["layer_norm"] = lambda: F.layer_norm(x, (H,), w_n, w_n, 1e-6)
    ops["silu"] = lambda: F.silu(x)
    ops["gelu"] = lambda: F.gelu(x)
    # 4. elementwise / reduction, cheap alone but they add up
    ops["add"] = lambda: x + x
    ops["mul"] = lambda: x * x
    ops["sum_last"] = lambda: x.sum(-1)
    ops["softmax"] = lambda: torch.softmax(x, dim=-1)
    # 5. the pair used by the contrastive losses
    ops["bmm_scores"] = lambda: torch.bmm(x[:, :128], x[:, :128].transpose(1, 2))
    ops["cross_entropy"] = lambda: F.cross_entropy(
        x.view(-1, H)[:4096].float(), torch.randint(0, H, (4096,), device=dev))

    out = {}
    for name, fn in ops.items():
        try:
            out[name] = bench(fn)
        except Exception as exc:
            out[name] = None
            print(f"  {name}: SKIP ({type(exc).__name__}: {str(exc)[:60]})")
    print(f"=== {args.label} ===")
    for name, ms in out.items():
        print(f"  {name:20} {ms:8.3f} ms" if ms else f"  {name:20} skip")
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
