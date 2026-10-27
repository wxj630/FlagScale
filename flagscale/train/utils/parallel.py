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


def iter_large_leaf_modules(
    module: nn.Module, min_bytes: int, gather_dtype: torch.dtype = torch.bfloat16
) -> Iterator[nn.Module]:
    """Yield leaf modules holding at least ``min_bytes`` of their own parameters.

    FSDP2 gathers a param group's parameters into one contiguous buffer, so a
    group that still contains giant embeddings needs a giant allocation.  On
    Qwen3-8B the root group is ``embed_tokens`` (151669x4096) plus ``lm_head``,
    which is 2.31 GiB in bf16 -- P800 fails that single allocation during the
    pre-backward all-gather even with 28 GiB of device memory free, because by
    then the free memory is scattered across segments.  Sharding these modules
    on their own halves the largest request so the allocator can satisfy it
    from a fragment it already has.

    Modules with shared (tied) parameters are skipped: sharding them separately
    would break the tie.
    """

    block_ids = {id(block) for block in iter_transformer_blocks(module)}

    ref_count: dict[int, int] = {}
    for p in module.parameters():
        ref_count[id(p)] = ref_count.get(id(p), 0) + 1

    def visit(current: nn.Module) -> Iterator[nn.Module]:
        for child in current.children():
            if id(child) in block_ids:
                continue
            own = list(child.parameters(recurse=False))
            has_nested = any(True for c in child.children() if any(True for _ in c.parameters()))
            if own and not has_nested:
                # The gathered buffer is in the compute dtype, not the master
                # dtype the weights are stored in.
                size = sum(p.numel() for p in own) * gather_dtype.itemsize
                tied = any(ref_count.get(id(p), 0) > 1 for p in own)
                if size >= min_bytes and not tied:
                    yield child
            else:
                yield from visit(child)

    yield from visit(module)


def fsdp2_available() -> bool:
    """Whether the installed torch exposes composable FSDP2."""
    try:
        from torch.distributed._composable.fsdp import fully_shard  # noqa: F401

        return True
    except Exception:
        return False


def _allgather_into_tensor_coalesced(self, output_tensors, input_tensors, opts=None):
    """Equivalent of the missing method: one ``_allgather_base`` per pair.

    ``allgather_coalesced`` exists on some vendor backends but raises
    NotImplementedError, so fall back to the per-tensor form.  FSDP2 only needs
    the data gathered; coalescing is a batching optimisation, not a semantic
    requirement.
    """

    if len(output_tensors) != len(input_tensors):
        raise ValueError(
            "expected matching output/input counts, got "
            f"{len(output_tensors)} and {len(input_tensors)}"
        )
    if opts is None:
        from torch._C import _distributed_c10d as _c10d

        opts = _c10d.AllgatherOptions()

    works = []
    for output_tensor, input_tensor in zip(output_tensors, input_tensors):
        works.append(self._allgather_base(output_tensor, input_tensor, opts))
    return works


def _process_group_classes() -> list[type]:
    """Backend process-group classes that may lack the coalesced all-gather.

    Keyed by the collective name each vendor backend registers.  FSDP2 picks the
    class from the live process group, so a patch has to land on the class the
    running backend actually uses:

    * CUDA / PPU (Torch-FL CUDA boxing) / Kunlunxin vendor torch -> NCCL
    * Ascend -> HCCL; ``torch_npu`` registers ``ProcessGroupHCCL`` and there is
      no ``ProcessGroupNCCL`` on that box at all, which is why a NCCL-only shim
      left FSDP2 unrunnable there.
    * Other vendors expose their own class or none; the funcol override below
      covers them regardless of which class the group belongs to.
    """

    classes: list[type] = []

    def _add(cls) -> None:
        if isinstance(cls, type) and cls not in classes:
            classes.append(cls)

    # NCCL: CUDA, PPU (Torch-FL CUDA boxing), Kunlunxin's vendor torch.
    try:
        from torch._C._distributed_c10d import ProcessGroupNCCL

        _add(ProcessGroupNCCL)
    except Exception:
        pass

    # HCCL: Ascend.  torch_npu owns the `npu` device and there is no
    # ProcessGroupNCCL on that box at all.  The class has moved between torch_npu
    # releases (`torch_npu._C._distributed_c10d`, and in some builds it is
    # re-exported from `torch.distributed.distributed_c10d`), so probe by
    # attribute instead of a single `from ... import` path.
    try:
        import torch_npu  # noqa: F401

        hccl_candidates = []
        c_ext = getattr(torch_npu, "_C", None)
        for holder in (
            getattr(c_ext, "_distributed_c10d", None) if c_ext is not None else None,
            getattr(c_ext, "distributed_c10d", None) if c_ext is not None else None,
            c_ext,
            torch_npu,
        ):
            if holder is not None:
                hccl_candidates.append(getattr(holder, "ProcessGroupHCCL", None))
        try:
            from torch.distributed.distributed_c10d import ProcessGroupHCCL as _PGH

            hccl_candidates.append(_PGH)
        except Exception:
            pass
        try:
            from torch._C._distributed_c10d import ProcessGroupHCCL as _PGH2

            hccl_candidates.append(_PGH2)
        except Exception:
            pass
        for candidate in hccl_candidates:
            if candidate is not None:
                _add(candidate)
    except Exception:
        pass

    try:
        from torch._C._distributed_c10d import ProcessGroupGloo

        _add(ProcessGroupGloo)
    except Exception:
        pass
    return classes


def install_coalesced_allgather_shim(log=lambda msg: None) -> bool:
    """Fill in `allgather_into_tensor_coalesced` when the vendor backend omits it.

    FSDP2 gathers a param group with `allgather_into_tensor_coalesced`. Some
    vendor builds ship ``allgather_coalesced`` (the list-of-lists form) but not
    the ``into_tensor`` spelling, so FSDP2 dies with

        AttributeError: 'ProcessGroupNCCL' object has no attribute
                        'allgather_into_tensor_coalesced'

    That is the entire difference measured on PPU: ``allgather_coalesced``
    exists, the tensor-view name does not. FSDP2 passes one (output, input)
    pair per param group, so the older API with a one-element outer list is the
    exact equivalent -- the batching it adds is what FSDP2 does not use.

    The patch goes onto every backend class this host exposes rather than only
    NCCL: on Ascend the group is a ``ProcessGroupHCCL`` and the NCCL class does
    not exist, so the earlier NCCL-only version could never have worked there.

    Preference order: fix it here rather than in Torch-FL, because this file is
    ours to change and the vendor gap is a calling-convention mismatch, not a
    missing capability.

    Returns True when a shim was installed, False when the method already
    exists or no equivalent is available.
    """

    classes = _process_group_classes()
    if not classes:
        log("coalesced all-gather shim: no backend process-group class importable")
        return False

    patched: list[str] = []
    skipped: list[str] = []
    for cls in classes:
        name = cls.__name__
        if hasattr(cls, "allgather_into_tensor_coalesced"):
            skipped.append(name)
            continue
        if not hasattr(cls, "_allgather_base"):
            skipped.append(f"{name}(no _allgather_base)")
            continue
        if not hasattr(cls, "allgather_into_tensor_coalesced"):
            cls.allgather_into_tensor_coalesced = _allgather_into_tensor_coalesced
            patched.append(name)
    if not patched:
        log(
            "coalesced all-gather shim: nothing to patch "
            f"({', '.join(skipped) or 'no classes'}); the backend already provides it"
        )
        return False
    log(
        "coalesced all-gather shim installed on "
        f"{', '.join(patched)}: per-tensor _allgather_base fallback"
    )

    # The patch above is not enough on its own: the `_c10d_functional` operators
    # are C++ with a statically registered CUDA kernel, so neither a Python
    # attribute on the process group nor `torch.library` IMPL can replace them
    # (`_dispatch_dump_table` shows CUDA bound to Functional.cpp). DTensor's
    # `_to_replicate_tensor` reaches the broken path through the *Python*
    # function `funcol.all_gather_tensor`, so that is what gets replaced.
    try:
        import torch.distributed._functional_collectives as _funcol

        def _group_object(group, tag):
            """Turn whatever DTensor passes into a ProcessGroup object."""
            if isinstance(group, torch.distributed.ProcessGroup):
                return group
            # A DeviceMesh is the common case here; it knows its own group.
            getter = getattr(group, "get_group", None)
            if callable(getter):
                try:
                    return getter()
                except Exception:
                    pass
            name = _funcol._resolve_group_name(group, tag)
            if str(name) in {"0", "world"}:
                return torch.distributed.group.WORLD
            raise ValueError(f"cannot resolve process group {name!r}")

        def _patched_all_gather_tensor(tensor, gather_dim, group, tag=""):
            if not tensor.is_contiguous():
                raise AssertionError("Tensor must be contiguous for all_gather_tensor")
            if gather_dim != 0:
                # The original only supports dim 0; keep that contract rather
                # than silently doing something different.
                raise NotImplementedError("all_gather_tensor shim supports gather_dim=0")
            pg = _group_object(group, tag)
            group_size = pg.size()
            out = torch.empty(
                tensor.numel() * group_size,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            # Call the process group directly: the public helper wraps the call in
            # a logging decorator that resolves `dist.get_backend(group)` by name,
            # which fails for the default group on this backend.
            from torch._C import _distributed_c10d as _c10d

            work = pg._allgather_base(
                out.view(group_size, *tensor.shape), tensor, _c10d.AllgatherOptions()
            )
            if hasattr(work, "wait"):
                work.wait()
            return torch.ops._c10d_functional.wait_tensor(out)

        _funcol.all_gather_tensor = _patched_all_gather_tensor
        log("funcol.all_gather_tensor overridden (per-tensor fallback for this backend)")
    except Exception as exc:  # pragma: no cover - backend/vendor dependent
        log(f"coalesced all-gather shim: funcol override skipped ({type(exc).__name__}: {exc})")

    return True


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
    reshard_after_forward: bool = True,
    max_group_bytes: int = 1 << 30,
    log=lambda msg: None,
) -> nn.Module:
    """Shard ``model`` in place with FSDP2 and return it.

    ``reshard_after_forward=True`` re-shards each block's parameters as soon as
    its forward finishes, so only the block being computed is ever unsharded.
    The GR00T trainer passes False (keep them gathered, skipping the backward
    all-gather), but that keeps *every* block's parameters resident at once --
    for an 8B model in fp32 that alone is ~30 GB per rank and overruns a 80 GB
    card. True costs one extra all-gather per block and fits.

    ``max_group_bytes`` caps how much bf16 weight one param group may gather at
    once.  Transformer blocks come in well under it, but a root group holding
    the token embedding and the output projection does not, and the resulting
    multi-gigabyte single allocation is what P800's allocator cannot satisfy
    once the free memory has fragmented.  Oversized leaf modules therefore get
    their own group, which halves the peak request.
    """

    from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh

    # Some vendor backends expose allgather_coalesced but not the into_tensor
    # spelling FSDP2 calls; install the equivalent before sharding starts.
    install_coalesced_allgather_shim(log=log)

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
    large_leaves = list(iter_large_leaf_modules(model, max_group_bytes, compute_dtype))
    names = {id(m): n for n, m in model.named_modules()}
    for leaf in large_leaves:
        nbytes = sum(p.numel() for p in leaf.parameters(recurse=False)) * compute_dtype.itemsize
        log(
            f"  oversized leaf: {names.get(id(leaf), '<unnamed>')} "
            f"gathers {nbytes / 2**20:.0f} MiB in {compute_dtype}"
        )
        fully_shard(
            leaf,
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
        f"FSDP2 sharded {len(blocks)} transformer blocks, {len(large_leaves)} "
        f"oversized leaf module(s) and the root over {world_size} rank(s); "
        f"compute dtype {compute_dtype}, reduce dtype {reduce_dtype}, "
        f"max param group {max_group_bytes / 2**20:.0f} MiB"
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

