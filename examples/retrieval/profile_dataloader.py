"""Time the train loader on its own: no model, no device work.

This separates "how fast can this machine produce batches" from "how fast can it
compute", which the wall-clock minus kernel-time subtraction cannot: that residual
also contains GPU idle waiting, so attributing it to the pipeline was a guess.
"""
import argparse, time
import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--conf", required=True)
ap.add_argument("--batches", type=int, default=200)
ap.add_argument("--workers", type=int, default=None, help="override num_workers")
a = ap.parse_args()

from flagscale.train.datasets.retrieval import build_split_loaders

cfg = OmegaConf.load(a.conf)
task = str(cfg.task)
data_cfg = cfg.data
if a.workers is not None:
    OmegaConf.set_struct(data_cfg, False)
    data_cfg.num_workers = a.workers

print(f"num_workers(config)={data_cfg.get('num_workers', 0)} batch_size={cfg.batch_size}", flush=True)
loaders = build_split_loaders(task, data_cfg, int(cfg.batch_size), pin_memory=False, seed=42)
train = loaders[0]

it = iter(train)
# warm up: first batches pay parquet open + arrow init
for _ in range(5):
    next(it)
t0 = time.perf_counter()
rows = 0
for _ in range(a.batches):
    b = next(it)
    rows += len(b["anchor"])
dt = time.perf_counter() - t0
print(f"dataloader only: {a.batches} batches, {rows} rows in {dt:.2f}s")
print(f"  {rows/dt:.0f} rows/s   {dt/a.batches*1000:.1f} ms/batch")
