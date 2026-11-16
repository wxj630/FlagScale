# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise retrieval FSDP2 backward, fp32 AdamW, and checkpoint gathering.

Run in the vendor environment with the repository on PYTHONPATH::

    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        examples/retrieval/smoke_fsdp2.py

This uses tiny weights; run the real reranker separately to validate capacity.
The one-row output weight deliberately tests uneven sharding/padding.
"""

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn

from flagscale.train.utils.parallel import apply_fsdp2, full_state_dict


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(128, 128), nn.Linear(128, 128)])
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return self.out(x)


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    device = torch.device("cuda", local_rank)
    probe = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(probe)
    assert probe.item() == world * (world + 1) / 2

    torch.manual_seed(42)
    model = Network().float()
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    model = apply_fsdp2(
        model,
        "cuda",
        world,
        log=lambda s: print(s, flush=True) if rank == 0 else None,
    )
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        x = torch.ones(4, 128, device=device) * (rank + 1) / world
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x).float().square().mean()
        assert torch.isfinite(loss).item()
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        norm_value = norm.full_tensor() if hasattr(norm, "full_tensor") else norm
        assert torch.isfinite(norm_value).item()
        optimizer.step()
        for param in model.parameters():
            assert param.dtype == torch.float32
            for key in ("exp_avg", "exp_avg_sq"):
                assert optimizer.state[param][key].dtype == torch.float32
        losses.append(loss.item())
        print(f"rank={rank} step={step} loss={loss.item()} fp32_state=True", flush=True)

    state = full_state_dict(model)
    if rank == 0:
        assert set(state) == set(initial)
        assert all(t.shape == initial[k].shape for k, t in state.items())
        assert all(torch.isfinite(t).all().item() for t in state.values())
        assert any(not torch.equal(state[k], initial[k]) for k in initial)
        restored = Network().float()
        restored.load_state_dict(state, strict=True)
        print(
            json.dumps({"status": "FSDP2_PROBE_PASS", "world_size": world, "losses": losses}),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
