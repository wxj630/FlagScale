"""Does use_flaggems actually change which kernels run?

The config key was previously a no-op (exported as an env var, read by nothing),
so it is not enough to check that the flag is set: this runs a few ops from the
real training path both ways and compares the underlying kernel names.

    python examples/retrieval/verify_flaggems.py \
        --conf examples/retrieval/conf/train/bge_m3.yaml --device cuda
"""
import argparse, os
import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--conf", required=True)
ap.add_argument("--device", default="auto")
a = ap.parse_args()

from flagscale.train.train_retrieval import init_distributed
from flagscale.train.utils.flaggems import enable_flaggems

os.environ.setdefault("FLAGSCALE_DEVICE", a.device)
rank, world, dev, platform = init_distributed(OmegaConf.create({"device_type": a.device}))
cfg = OmegaConf.load(a.conf)
hardware = cfg.get("hardware", {})

x = torch.randn(64, 512, dtype=torch.float32, device=dev)
w = torch.randn(512, dtype=torch.float32, device=dev)


def op_kernels():
    """Every device kernel name observed for one pass of the test ops.

    The filter cannot look for "triton" in the name: on Kunlunxin FlagGems'
    Triton kernels are compiled ahead of time and surface as ordinary vendor
    kernel symbols, so the diff of the whole name set is what shows the swap.
    """
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.nn.functional.layer_norm(x, (512,), w, w, 1e-6)
        torch.nn.functional.gelu(x)
        torch.add(x, x)
        torch.sum(x, dim=-1)
        _ = torch.mm(x, x.T)
        torch.cuda.synchronize()
    return {e.key for e in prof.key_averages() if e.self_device_time_total > 0}


before = op_kernels()
print(f"before: {len(before)} device kernels")

enabled = enable_flaggems(hardware, dev, platform)
if not enabled:
    # Force it on so the probe still answers "would it work here?".
    print("use_flaggems is false in the config; forcing it on for the probe")
    hardware = OmegaConf.create({"use_flaggems": True, "flaggems_required": True})
    enabled = enable_flaggems(hardware, dev, platform)

after = op_kernels()
print(f"after:  {len(after)} device kernels")
print()
print("only with flag_gems:")
for n in sorted(after - before)[:8]:
    print("  +", n[:92])
print("only without:")
for n in sorted(before - after)[:8]:
    print("  -", n[:92])
print()
print(f"RESULT: kernel set changed = {after != before} "
      f"({len(before - after)} gone, {len(after - before)} new)")
