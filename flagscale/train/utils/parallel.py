# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Parallelism helpers for the native retrieval trainer.

Two strategies are supported, selected by ``train.hardware.parallel``:

``ddp`` (default)
    Classic DistributedDataParallel.  Every rank holds a full replica; the
    optimizer keeps whatever dtype the parameters have.  This is what the
    existing single-GPU runs used, and it stays the default so checkpoint
    compatibility and vendor coverage do not change underneath the experiments.

``fsdp2``
    Composable fully-sharded data parallel (``torch.distributed._composable.fsdp``).
    Parameters, gradients and optimizer state are sharded across ranks, and the
    model is kept in ``master_dtype`` (fp32 by default) while parameters are
    cast to the compute dtype only for the forward/backward of each block.  This
    is the shape the GR00T trainer uses, and it is what makes full-parameter
    AdamW on an 8B model fit: the 16-bytes-per-parameter budget is divided by
    the world size instead of being replicated per card.

Why the master dtype matters
    With ``ddp`` the optimizer states inherit the parameter dtype.  When the
    parameters are bf16 (as the retrieval configs load them), AdamW's
    ``exp_avg``/``exp_avg_sq`` end up bf16 too, and a learning rate of 1e-5
    produces updates below the 2^-8 relative resolution of bf16 -- the update is
    silently rounded away.  ``fsdp2`` keeps params (and therefore states) in
    fp32 and only downcasts for compute, which is the standard mixed-precision
    contract.

Checkpointing
    FSDP2 parameters are DTensors.  ``state_dict`` on a sharded module yields
    shards, so saving must go through
    ``get_model_state_dict(..., StateDictOptions(full_state_dict=True,
    cpu_offload=True))``, which is a collective that every rank must call.
    The gathered tensors are plain ``torch.Tensor`` and can be loaded into an
    unsharded Hugging Face model to write the usual ``save_pretrained`` layout
    that the evaluation entrypoint consumes.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch
from torch import nn

# The adapter wraps the Hugging Face backbone (``model.backbone``), so block
# lists sit at e.g. ``backbone.encoder.layer`` for BERT-style encoders and
# ``backbone.vision_model.encoder.layers`` for CLIP.  Rather than hardcoding a
# path per family, discovery walks every submodule and keeps the repeated
# block lists, which is what makes block-wise sharding possible: FSDP2 can then
# release each block's gathered parameters before the next one runs, instead of
# gathering the whole backbone at once.


def _is_block_list(module: nn.Module) -> bool:
    return isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) > 1


def fsdp2_available() -> bool:
    """Whether the installed torch exposes composable FSDP2."""
    try:
        from torch.distributed._composable.fsdp import fully_shard  # noqa: F401

        return True
    except Exception:
        return False


def iter_transformer_blocks(module: nn.Module) -> Iterator[nn.Module]:
    """Yield the members of every repeated-block list, deepest first.

    Multiple towers (CLIP's text and vision encoders) are both covered.  An
    empty iterator means no block list was found; :func:`apply_fsdp2` then
    shards the root module only, which is correct but coarser.
    """

    found: list[nn.ModuleList] = []

    def walk(current: nn.Module) -> None:
        for child in current.children():
            if _is_block_list(child):
                found.append(child)
            walk(child)

    walk(module)
    if not found:
        return
    # Deepest (shortest) containers last so inner blocks are sharded first.
    found.sort(key=len)
    for block_list in found:
        for block in reversed(list(block_list)):
            yield block


def apply_fsdp2(
    model: nn.Module,
    device_type: str,
    world_size: int,
    compute_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.bfloat16,
    reshard_after_forward: bool = False,
    log=lambda msg: None,
) -> nn.Module:
    """Shard ``model`` in place with FSDP2 and return it.

    ``reshard_after_forward=False`` mirrors the GR00T trainer: parameters stay
    gathered between forward and backward so the recompute pass during
    backward does not all-gather them a second time.
    """

    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh(device_type, (world_size,))
    mp_policy = MixedPrecisionPolicy(
        param_dtype=compute_dtype,
        reduce_dtype=reduce_dtype,
    )

    blocks = list(iter_transformer_blocks(model))
    for block in blocks:
        fully_shard(
            block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    log(
        f"FSDP2 sharded {len(blocks)} transformer blocks + root over "
        f"{world_size} rank(s); compute dtype {compute_dtype}, "
        f"reduce dtype {reduce_dtype}"
    )
    return model


@contextlib.contextmanager
def gradient_sync_disabled(model: nn.Module) -> Iterator[None]:
    """Skip gradient all-reduce for one micro-step, restoring on exit.

    DDP exposes ``no_sync()``; FSDP2 exposes ``set_requires_gradient_sync``.
    Both matter: without this, ``gradient_accumulation_steps=N`` performs N
    collectives where one would do, which is the difference between usable and
    unusable throughput once the ranks live on different hosts.
    """

    ddp_no_sync = getattr(model, "no_sync", None)
    if callable(ddp_no_sync):
        with ddp_no_sync():
            yield
        return

    set_sync = getattr(model, "set_requires_gradient_sync", None)
    if callable(set_sync):
        set_sync(False)
        try:
            yield
        finally:
            set_sync(True)
        return

    yield


def full_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Gather a full (unsharded) state dict of plain CPU tensors.

    Collective: all ranks must call this.  ``StateDictOptions`` asks for a full
    state, but with composable FSDP2 the values can still be DTensors, whose
    local shard has been materialised into a wrapper without a usable storage
    pointer; ``safetensors`` needs a real one, so unwrap to ``to_local()``.
    """

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    gathered = get_model_state_dict(model, options=options)
    return {k: _as_plain_tensor(v) for k, v in gathered.items()}


def _as_plain_tensor(value: Any) -> Any:
    """Unwrap a DTensor (or nested container of them) into plain CPU tensors.

    ``to_local()`` on an offloaded DTensor can hand back a view whose storage is
    already released, which ``safetensors`` rejects when it asks for the data
    pointer.  Materialise a fresh contiguous CPU copy rather than relying on the
    wrapper's storage.
    """

    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        local = to_local()
        if torch.is_tensor(local):
            return local.detach().to("cpu").contiguous().clone()
        return _as_plain_tensor(local)
    if torch.is_tensor(value):
        return value.detach().to("cpu").contiguous().clone()
    if isinstance(value, dict):
        return {k: _as_plain_tensor(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_as_plain_tensor(v) for v in value)
    return value


def full_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> dict:
    """Gather optimizer state for resumable checkpoints.  Collective as well."""

    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_optimizer_state_dict(model, optimizer, options=options)
