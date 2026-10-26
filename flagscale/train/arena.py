# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""REJECTED: single-arena allocator for devices without ``expandable_segments``.

===========================  DO NOT WIRE THIS UP  ===========================
This module is kept as the record of a measured dead end, not as working code.
Every route it takes was tested on P800 and failed. The evidence is in
``Notes/kunlunxin-alloc-fragmentation.md`` (sections 3.1 and 3.2); the two
blockers, in short:

1. **MemPool + private engine-thread binding aborts the process at exit.** The
   binding itself works -- the probe confirms the FSDP2 all-gather buffer does
   land in the pool. But the pool stays referenced by the autograd engine
   thread, so ``~MemPool`` trips ``use_count() == 1 INTERNAL ASSERT FAILED``
   on interpreter shutdown (exit 1/134). Re-binding the same thread also raises
   ``beginAllocateToPool: already recording to mempool_id``.

2. **Multi-stream defeats the pre-reserve.** PyTorch's caching allocator pools
   per stream and FSDP2's all-gather uses a dedicated one, so the single 70 GiB
   segment is replaced by 20 per-stream segments the moment other streams
   allocate:

       arena reserved+free      segs= 1 total=70.00 free=70.00 largest=70.00
       after churn (3 streams)  segs=20 total=28.02 free=28.02 largest= 2.31

   Measured end to end: with ``FLAGSCALE_ARENA_GIB=80`` set, the failing
   allocation moved from 2.31 GiB to 1.16 GiB (the post-shard all-gather size)
   and still OOM'd.

The ``never release the pool`` trick in :func:`enable` does not fix (1); it only
defers the abort to process teardown.

What does work today: shard the oversized leaves into their own param group
(``iter_large_leaf_modules`` in ``utils/parallel.py``), and/or run the reranker
on 3 cards. The real fix is a pluggable allocator that takes one contiguous
region via ``cuMemAlloc`` and sub-allocates from it -- NOT implemented.
============================================================================

Original description follows, for context only.

CUDA's ``expandable_segments`` keeps one virtual address range and maps physical
pages into it on demand, so a segment can grow and every allocation is carved
from the same region.  Kunlunxin's P800 has no such option, and the vendor SDK
cannot be made to support it: ``cuMemAddressReserve`` / ``cuMemCreate`` return
``CUDA_ERROR_NOT_SUPPORTED``.  What the device does instead is worse than plain
fragmentation -- ``cuMemAlloc`` fails for 4 GiB while 15.6 GiB is reported free,
because the physical pages are already scattered and the driver will not
compact them.  The reranker hits this as a 2.31 GiB request inside the fused
linear backward with 23 GiB nominally available.

This module reproduces the useful half of ``expandable_segments``: reserve one
large contiguous block at startup, hand it back to the allocator's free list,
and route every later allocation into it through :class:`torch.cuda.MemPool`.
Inside one segment PyTorch splits and coalesces normally, so the arena behaves
like an expandable segment as long as the peak fits in it.

Routing has to cover two threads.  ``use_mem_pool`` is thread-local, and FSDP2's
pre-backward all-gather runs on autograd's persistent engine thread, so
:func:`bind_now` runs a one-node backward to bind that thread once per process.

Opt-in: nothing happens unless ``FLAGSCALE_ARENA_GIB`` is set.
"""

from __future__ import annotations

import gc
import os

import torch


GIB = 1 << 30


def supported() -> bool:
    needed = ("MemPool", "_cuda_beginAllocateCurrentThreadToPool")
    return all(hasattr(torch.cuda.memory, name) for name in needed)


def requested_gib() -> float:
    """Arena size from ``FLAGSCALE_ARENA_GIB``; 0 disables the feature."""
    try:
        return max(0.0, float(os.environ.get("FLAGSCALE_ARENA_GIB", "0")))
    except ValueError:
        return 0.0


class _BindPool(torch.autograd.Function):
    """No-op autograd node that binds the engine thread when it runs."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, pool_id) -> torch.Tensor:
        ctx.pool_id = pool_id
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        try:
            torch.cuda.memory._cuda_beginAllocateCurrentThreadToPool(
                torch.cuda.current_device(), ctx.pool_id
            )
        except Exception:
            pass
        return grad_output, None


def bind_now(pool) -> bool:
    """Bind autograd's engine thread for this device to ``pool``.

    Must run before the first real backward: FSDP2 allocates its pre-backward
    all-gather buffer inside the engine thread, and a binding installed later
    would only affect the next step.  A one-node graph is enough -- the engine
    keeps one worker thread per device, and the binding lives on the thread.
    """

    if pool is None:
        return False
    probe = torch.zeros(1, device="cuda", requires_grad=True)
    try:
        _BindPool.apply(probe, pool.id).backward()
        return True
    except Exception:
        return False


def enable(gib: float | None = None, log=lambda msg: None):
    """Reserve the arena, bind this thread and the engine thread.

    Returns the pool (``None`` when disabled or unsupported).  The pool is
    deliberately never released: the autograd thread stays bound to it for the
    life of the process, and tearing it down would trip ``~MemPool``'s
    ``use_count == 1`` assertion at interpreter shutdown.

    REJECTED -- see the module docstring.  This still aborts at process exit and
    is still defeated by multiple streams; the warning below exists so nobody
    wires it up without seeing that.
    """

    gib = requested_gib() if gib is None else gib
    if gib <= 0:
        return None
    log(
        "WARNING: FLAGSCALE_ARENA_GIB is set, but the arena route was measured "
        "dead on P800 -- MemPool aborts at interpreter exit (use_count==1) and "
        "multi-stream allocation defeats the pre-reserve. See "
        "Notes/kunlunxin-alloc-fragmentation.md sections 3.1/3.2."
    )
    if not supported():
        log("arena allocator requested but this torch build lacks MemPool; using the default allocator")
        return None

    memory = torch.cuda.memory
    device_index = torch.cuda.current_device()
    pool = memory.MemPool()
    memory._cuda_beginAllocateCurrentThreadToPool(device_index, pool.id)
    reserve = torch.empty(int(gib * GIB), dtype=torch.uint8, device="cuda")
    del reserve
    gc.collect()
    segments = pool.snapshot()
    log(
        "arena allocator: reserved %.1f GiB as %d segment(s) on device %d"
        % (gib, len(segments), device_index)
    )
    if bind_now(pool):
        log("arena allocator: autograd engine thread bound to the pool")
    return pool


def describe() -> str:
    """One-line summary for the training log."""
    gib = requested_gib()
    if gib <= 0:
        return "arena allocator: disabled"
    if not supported():
        return "arena allocator: requested but unsupported by this torch build"
    return f"arena allocator: {gib:.1f} GiB requested (REJECTED route, see module docstring)"
