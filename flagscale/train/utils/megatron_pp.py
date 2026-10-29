"""Pipeline-parallel training driver for retrieval models on Megatron-LM-FL.

This is the ``train.parallel=megatron`` backend of :mod:`flagscale.train.
train_retrieval`.  It owns only what pipeline parallelism changes:

* the pipeline schedule (``forward_backward_func``) instead of plain
  forward/backward,
* Megatron's sharded distributed optimizer instead of a replicated one,
* per-stage checkpoint files instead of a single HF ``model.safetensors``.

Everything else stays on the FlagScale side and is passed in: the datasets and
loaders (so the query-level split is the same one DDP/FSDP2 use), the prompt
framing (from :mod:`flagscale.models.retrieval.qwen3_reranker`), the epoch loop's
checkpointing policy (``checkpoint-epoch-{N}``), wandb, and ``metrics.json``.

Why not FSDP2 on these vendors: FSDP2 gathers every param group into one
contiguous buffer per rank.  ``lm_head`` is 151669x4096, so that buffer is
2.31 GiB in fp32 and does **not** shrink with more cards; on P800 the allocator
cannot satisfy it once free memory has fragmented (no ``expandable_segments``).
Pipeline parallelism never materialises the whole model on one rank, and the
distributed optimizer still shards the fp32 master weights and both Adam
moments, so the 16-bytes-per-parameter layout the A100 baseline uses is
recovered.
"""

from __future__ import annotations

import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F

from flagscale.models.retrieval.qwen3_reranker_megatron import (
    build_megatron_config,
    build_megatron_model,
    build_prompt_body,
    load_stage_weights,
    prompt_prefix,
    prompt_suffix,
    read_hf_config,
)


def _log_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        print(message, flush=True)


def _last_stage() -> bool:
    """Only the last pipeline stage produces the loss.

    Logging from global rank 0 (the first stage) averages an empty list and
    prints ``nan``, so every loss-bearing message is gated on this.
    """

    return dist.get_rank() == dist.get_world_size() - 1


def _init_model_parallel(pp: int, tp: int, backend: str, seed: int) -> tuple[int, int, int]:
    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(torch.cuda.current_device())

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp
        )
    model_parallel_cuda_manual_seed(seed=seed)
    return rank, world_size, local_rank


class _PairBatcher:
    """Turn FlagScale's text rows into padded (query, document) token blocks.

    Tokenisation happens per batch rather than through an on-disk cache: one
    step is ~20 s of 8B compute on these vendors, while encoding 128 short
    sequences takes tens of milliseconds.  Dropping the cache removes a second
    dataset implementation, so the split can only ever come from FlagScale's.

    ``pad_to`` is not optional on the vendor runtimes: they compile kernels per
    shape, so a batch-max padding width that changes every step recompiles every
    step and the run appears to hang.
    """

    def __init__(
        self,
        tokenizer,
        *,
        max_length: int,
        instruction: str,
        score_token_ids: tuple[int, int],
        pad_to: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.instruction = instruction
        self.score_token_ids = torch.tensor(list(score_token_ids), dtype=torch.long)
        self.pad_to = int(pad_to or max_length)
        self.device = device

        prefix = prompt_prefix()
        suffix = prompt_suffix()
        self.prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        self.suffix_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
        self.budget = max(16, self.max_length - len(self.prefix_ids) - len(self.suffix_ids))
        # Right padding needs a pad token; Qwen uses its eos token.
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        if pad_id is None:
            raise RuntimeError("tokenizer has neither pad_token_id nor eos_token_id")
        self.pad_id = int(pad_id)

    def encode(self, pairs: list[list[str]]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        bodies = [build_prompt_body(self.instruction, q, d) for q, d in pairs]
        encoded = self.tokenizer(
            bodies,
            add_special_tokens=False,
            truncation=True,
            max_length=self.budget,
        )["input_ids"]
        rows = [self.prefix_ids + ids + self.suffix_ids for ids in encoded]

        width = self.pad_to
        for row in rows:
            if len(row) > width:
                raise RuntimeError(
                    f"encoded row of {len(row)} tokens exceeds pad_to={width}; "
                    "raise train.model.max_length or lower it consistently"
                )
        input_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        lengths = torch.zeros(len(rows), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row, dtype=torch.long)
            lengths[index] = len(row)

        position_ids = (
            torch.arange(width, dtype=torch.long).unsqueeze(0).expand(len(rows), width).contiguous()
        )
        inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": None,
        }
        context = {
            "lengths": lengths,
            "score_token_ids": self.score_token_ids,
            "rows": len(rows),
        }
        if self.device is not None:
            inputs = {k: (v.to(self.device) if v is not None else None) for k, v in inputs.items()}
            context["lengths"] = context["lengths"].to(self.device)
            context["score_token_ids"] = context["score_token_ids"].to(self.device)
        return inputs, context


def _score_processor(*, context, hidden_states, output_layer, **kwargs):
    """Score each row from the last real position's yes/no logits.

    Right padding plus a causal mask means a real token's hidden state never
    attends a pad, so ``lengths - 1`` indexes the true final prompt token.

    The axis order of ``hidden_states`` differs between Megatron versions (this
    fork hands the hook ``[b, s, h]`` while the stock NVIDIA model documents
    ``[s, b, h]``), so the sequence axis is identified by matching it against
    the row count instead of being assumed.

    The last position is selected by a one-hot mask rather than advanced
    indexing: ``hidden[positions, arange]`` faults inside the vendor XPytorch
    index kernel on P800.
    """

    lengths = context["lengths"]
    score_ids = context["score_token_ids"]
    rows = int(context["rows"])

    if hidden_states.size(0) == rows:
        hidden = hidden_states.transpose(0, 1).float()  # [b, s, h] -> [s, b, h]
    elif hidden_states.size(1) == rows:
        hidden = hidden_states.float()
    else:
        raise RuntimeError(
            f"hidden_states {tuple(hidden_states.shape)} matches neither axis to rows={rows}"
        )

    width = hidden.size(0)
    if int(lengths.max()) > width:
        raise RuntimeError(f"length {int(lengths.max())} exceeds padded width {width}")

    seq = torch.arange(width, device=hidden.device, dtype=torch.long)
    selector = (seq.unsqueeze(1) == (lengths - 1).unsqueeze(0)).float().unsqueeze(-1)
    last_hidden = (hidden * selector).sum(dim=0)

    # Two rows of the LM head, not the 151669-row vocabulary projection.
    head = output_layer.weight.index_select(0, score_ids).float()
    logits = last_hidden @ head.t()
    scores = logits[:, 0] - logits[:, 1]

    labels = context["labels"]
    loss = F.binary_cross_entropy_with_logits(scores, labels.float())
    with torch.no_grad():
        predicted = (scores > 0).float()
        context["scores"] = scores.detach()
        context["metrics"] = {
            "loss": float(loss.detach()),
            "accuracy": float((predicted == labels.float()).float().mean()),
            "rows": rows,
        }
    return loss


def _loss_reducer(context, output_tensor):
    metrics = context.get("metrics") or {}
    device = output_tensor.device
    return output_tensor, {
        "loss": torch.tensor(float(metrics.get("loss", 0.0)), device=device),
        "accuracy": torch.tensor(float(metrics.get("accuracy", 0.0)), device=device),
        "rows": torch.tensor(float(metrics.get("rows", 0)), device=device),
    }


class _RowStream(Iterator[list[list[str]]]):
    """Serve micro-batches of (query, document) pairs to the pipeline schedule.

    ``rows`` scopes the stream to the caller's split (train / validation / test)
    so the pipeline never trains on held-out data.  Iteration ends only when the
    caller has taken its planned number of steps: raising StopIteration inside
    ``forward_step`` would leave other ranks waiting on the schedule, which
    hangs instead of ending the epoch.
    """

    def __init__(self, loader, pairs_per_row: int, rows_per_microbatch: int) -> None:
        self.loader = loader
        self.pairs_per_row = int(pairs_per_row)
        self.rows_per_microbatch = int(rows_per_microbatch)
        self.microbatches_per_step = 1
        self.buffer: list[list[str]] = []
        self.buffer_rows = 0

    def reset(self) -> None:
        self.loader = iter(self.loader)
        self.buffer = []
        self.buffer_rows = 0

    def _fill(self, want_rows: int) -> None:
        while self.buffer_rows < want_rows:
            try:
                batch = next(self.loader)
            except StopIteration:
                self.loader = iter(self.loader)
                batch = next(self.loader, None)
                if batch is None:
                    raise RuntimeError("training loader produced no rows") from None
            anchors = batch["anchor"]
            for index, anchor in enumerate(anchors):
                pair_list = [[anchor, batch["positive"][index]]]
                for key in sorted(k for k in batch if k.startswith("negative")):
                    pair_list.append([anchor, batch[key][index]])
                if len(pair_list) != self.pairs_per_row:
                    raise RuntimeError(
                        f"row expands to {len(pair_list)} pairs but config says "
                        f"{self.pairs_per_row}"
                    )
                self.buffer.extend(pair_list)
                self.buffer_rows += 1

    def __iter__(self) -> "_RowStream":
        return self

    def __next__(self) -> list[list[str]]:
        self._fill(self.rows_per_microbatch)
        take = self.rows_per_microbatch * self.pairs_per_row
        out, self.buffer = self.buffer[:take], self.buffer[take:]
        self.buffer_rows -= self.rows_per_microbatch
        return out


def count_split_rows(
    path: str,
    split: str,
    *,
    split_modulo: int = 20,
    max_samples: int | None = None,
    log: Callable[[str], None] = _log_rank0,
) -> int:
    """Rows of ``split`` in ``path``, using FlagScale's own split rule.

    The loaders are streaming ``IterableDataset``s with no ``__len__``, but the
    pipeline schedule needs a step budget up front: it cannot be driven by
    exhausting an iterator, because a rank that runs out early stops pulling and
    the other stages wait on it forever.  Only the ``anchor`` column is read, so
    this is a cheap scan rather than a full pass over the corpus.
    """

    from flagscale.train.datasets.retrieval import split_accepts

    root = Path(path)
    files = [root] if root.is_file() else sorted(root.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {path}")

    import pyarrow.parquet as pq

    seen = 0
    accepted = 0
    for file_path in files:
        parquet = pq.ParquetFile(file_path)
        for batch in parquet.iter_batches(columns=["anchor"], batch_size=4096):
            for anchor in batch.column("anchor").to_pylist():
                row_index = seen
                seen += 1
                if split_accepts(row_index, split, split_modulo, group_key=str(anchor or "")):
                    accepted += 1
                    if max_samples is not None and accepted >= max_samples:
                        log(f"split={split} rows={accepted} (capped by max_samples)")
                        return accepted
    log(f"split={split} rows={accepted}")
    return accepted


class MegatronPipelineTrainer:
    """Pipeline-parallel trainer for one retrieval model.

    ``train_epoch`` returns the mean loss; checkpoint writing stays with the
    caller so the on-disk layout (``checkpoint-epoch-{N}``) is identical to the
    DDP/FSDP2 runs.
    """

    def __init__(
        self,
        *,
        model_path: str,
        stage_dir: str,
        pp: int,
        tp: int,
        micro_batch_rows: int,
        grad_accum: int,
        max_length: int,
        pairs_per_row: int,
        score_token_ids: tuple[int, int],
        instruction: str,
        lr: float,
        min_lr: float,
        lr_decay_style: str,
        warmup_steps: int,
        weight_decay: float,
        clip_grad: float,
        seed: int,
        device: torch.device,
        backend: str,
        attention_backend: str = "flash",
        recompute: bool = True,
        skip_nonfinite: bool = False,
        log_every: int = 1,
        log: Callable[[str], None] = _log_rank0,
    ) -> None:
        from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

        self.pp = int(pp)
        self.tp = int(tp)
        self.max_length = int(max_length)
        self.pairs_per_row = int(pairs_per_row)
        self.grad_accum = max(1, int(grad_accum))
        self.micro_batch_rows = int(micro_batch_rows)
        self.device = device
        self.log = log
        self.skip_nonfinite = bool(skip_nonfinite)
        self._log_every = max(1, int(log_every))
        self.last_epoch_stats: dict[str, float] = {}

        from transformers import AutoTokenizer

        _init_model_parallel(self.pp, self.tp, backend, seed)
        from megatron.core import parallel_state

        self.world_size = int(dist.get_world_size())
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.dp_rank = parallel_state.get_data_parallel_rank()
        self.dp_size = parallel_state.get_data_parallel_world_size()

        self.hf = read_hf_config(model_path)
        config = build_megatron_config(
            self.hf,
            pipeline_model_parallel_size=self.pp,
            tensor_model_parallel_size=self.tp,
            micro_batch_size=self.micro_batch_rows,
            recompute=recompute,
            attention_backend=attention_backend,
        )
        model = build_megatron_model(
            config,
            self.hf,
            pre_process=self.pp_rank == 0,
            post_process=self.pp_rank == self.pp - 1,
            seq_length=self.max_length,
        )
        loaded = load_stage_weights(
            model, Path(stage_dir) / f"stage{self.pp_rank}.pt", self.pp_rank
        )
        model.to(device)
        log(f"stage{self.pp_rank}: loaded {loaded} tensors from {stage_dir}")

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            use_distributed_optimizer=True,
            check_for_nan_in_grad=False,
        )
        self.model = DistributedDataParallel(
            config=config, ddp_config=ddp_config, module=model
        )
        self.models = [self.model]

        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=lr,
            min_lr=min_lr,
            weight_decay=weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_eps=1e-8,
            clip_grad=clip_grad,
            bf16=True,
            fp16=False,
            use_distributed_optimizer=True,
            params_dtype=torch.bfloat16,
            # fp32 master weights plus both moments in fp32: this is the
            # 16-bytes-per-parameter layout FSDP2 cannot reach on P800.
            use_precision_aware_optimizer=False,
        )
        self.optimizer = get_megatron_optimizer(optimizer_config, self.models)

        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        self._lr = float(lr)
        self._min_lr = float(min_lr)
        self._decay_style = str(lr_decay_style)
        self._warmup = int(warmup_steps)
        self._weight_decay = float(weight_decay)
        self.scheduler = self._make_scheduler(self._warmup)

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        self.batcher = _PairBatcher(
            tokenizer,
            max_length=self.max_length,
            instruction=instruction,
            score_token_ids=score_token_ids,
            pad_to=self.max_length,
            device=device,
        )
        self.pairs_per_microbatch = self.micro_batch_rows * self.pairs_per_row * self.dp_size
        self.steps_done = 0

    def _make_scheduler(self, decay_steps: int):
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

        return OptimizerParamScheduler(
            self.optimizer,
            init_lr=(self._lr * 0.01) if self._warmup > 0 else self._lr,
            max_lr=self._lr,
            min_lr=self._min_lr,
            lr_warmup_steps=self._warmup,
            # Never zero: Megatron divides by this.  Re-anchored after epoch 1
            # once the real epoch length has been measured.
            lr_decay_steps=max(1, int(decay_steps)),
            lr_decay_style=self._decay_style,
            start_wd=self._weight_decay,
            end_wd=self._weight_decay,
            wd_incr_steps=max(1, int(decay_steps)),
            wd_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=True,
        )

    def reanchor_schedule(self, total_steps: int) -> None:
        """Point the cosine at the measured total step count."""
        total = max(1, int(total_steps))
        self.scheduler = self._make_scheduler(total)
        self.log(
            f"lr schedule anchored: {self._decay_style} {self._lr:g} -> "
            f"{self._min_lr:g} over {total} steps, warmup {self._warmup}"
        )

    def current_lr(self) -> float:
        return float(self.scheduler.get_lr(self.optimizer.param_groups[0]))

    def _forward_step(
        self, data_iterator, model, *, score_token_ids, pad_to
    ):
        try:
            pairs = next(data_iterator)
        except StopIteration:
            return None
        inputs, context = self.batcher.encode(pairs)
        context["labels"] = torch.tensor(
            [1.0 if index % self.pairs_per_row == 0 else 0.0 for index in range(len(pairs))],
            device=inputs["input_ids"].device,
        )
        output = model(
            input_ids=inputs["input_ids"],
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            output_processor=_score_processor,
            output_processor_context=context,
        )
        return output, partial(_loss_reducer, context)

    def train_epoch(self, loader, epoch: int, epochs: int, steps: int) -> dict[str, float]:
        """Run ``steps`` optimizer steps of pipeline-parallel training.

        ``steps`` comes from the caller's row count, not from draining the
        loader: the pipeline schedule pulls micro-batches from an iterator, and
        an iterator that ends early on one rank leaves the other stages blocked
        in a send/recv with nobody to pair with.
        """

        from megatron.core.distributed.finalize_model_grads import finalize_model_grads
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        self.model.train()
        schedule = get_forward_backward_func()
        stream = _RowStream(loader, self.pairs_per_row, self.micro_batch_rows * self.dp_size)
        stream.reset()

        started = time.time()
        accumulated = 0.0
        logged = 0
        log_every = max(1, int(self._log_every))

        for _ in range(int(steps)):
            self.optimizer.zero_grad()
            losses = schedule(
                forward_step_func=partial(self._forward_step, score_token_ids=None, pad_to=self.max_length),
                data_iterator=[stream] * len(self.models),
                model=self.models,
                num_microbatches=self.grad_accum,
                seq_length=self.max_length,
                micro_batch_size=self.pairs_per_microbatch,
                decoder_seq_length=self.max_length,
                forward_only=False,
            )

            values = [
                float(entry["loss"])
                for entry in losses
                if isinstance(entry, dict) and "loss" in entry
            ]
            step_loss = sum(values) / len(values) if values else float("nan")

            if self.skip_nonfinite:
                # A bad loss is broadcast so every rank skips the same update;
                # ranks that stepped while others did not would desynchronise
                # the weights immediately.
                bad_local = bool(values) and (
                    any(v != v for v in values) or any(v > 1e3 for v in values)
                )
                flag = torch.tensor([1.0 if bad_local else 0.0], device=self.device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if float(flag) > 0.0:
                    self.log(f"step {self.steps_done + 1}: non-finite loss {values}, skipping update")
                    self.optimizer.zero_grad()
                    self.steps_done += 1
                    continue

            finalize_model_grads(self.models)
            # MegatronOptimizer.step() clips internally when config.clip_grad is
            # set and reports the pre-clip norm, so clipping again here would
            # scale the gradients twice.
            update_ok, grad_norm, _ = self.optimizer.step()
            self.scheduler.step(1)
            self.steps_done += 1

            if update_ok and values:
                accumulated += step_loss
                logged += 1

            if logged and logged % log_every == 0 and _last_stage():
                elapsed = time.time() - started
                pairs_done = logged * self.pairs_per_microbatch * self.grad_accum
                print(
                    f"epoch={epoch}/{epochs} step={self.steps_done} loss={step_loss:.4f} "
                    f"grad_norm={float(grad_norm) if grad_norm is not None else float('nan'):.3f} "
                    f"lr={self.current_lr():.2e} "
                    f"pairs/s={pairs_done / max(elapsed, 1e-6):.1f} elapsed={elapsed:.0f}s",
                    flush=True,
                )

        mean_loss = accumulated / logged if logged else float("nan")
        self.last_epoch_stats = {
            "loss": mean_loss,
            "steps": self.steps_done,
            "elapsed_s": time.time() - started,
        }
        return self.last_epoch_stats

    def evaluate(self, loader, eval_batch_rows: int = 4) -> dict[str, float]:
        """Loss/accuracy over ``loader`` through the pipeline schedule.

        Calling ``model(...)`` directly on each rank would run only that rank's
        slice of the layers and silently score from a fragment of the model, so
        evaluation goes through ``forward_backward_func(forward_only=True)``.

        Every rank reaches the collectives at the end even when it saw no rows:
        only the last stage produces scores, so the others accumulate zero, and
        an early return would let one rank skip an all-reduce the others enter.
        """

        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

        self.model.eval()
        schedule = get_forward_backward_func()
        total_loss = 0.0
        total_correct = 0.0
        total_rows = 0.0
        for batch in loader:
            anchors = batch["anchor"]
            pairs: list[list[str]] = []
            for index, anchor in enumerate(anchors):
                pairs.append([anchor, batch["positive"][index]])
                for key in sorted(k for k in batch if k.startswith("negative")):
                    pairs.append([anchor, batch[key][index]])

            # Chunk by row count so a full split does not need one enormous
            # forward; forward-only has no optimizer state, so a wider batch is
            # cheap but still bounded by the smallest card.
            step_pairs = max(1, int(eval_batch_rows)) * self.pairs_per_row
            for start in range(0, len(pairs), step_pairs):
                chunk = pairs[start : start + step_pairs]
                inputs, context = self.batcher.encode(chunk)
                context["labels"] = torch.tensor(
                    [1.0 if i % self.pairs_per_row == 0 else 0.0 for i in range(len(chunk))],
                    device=inputs["input_ids"].device,
                )
                stream = iter([chunk])
                with torch.no_grad():
                    outputs = schedule(
                        forward_step_func=partial(
                            self._forward_step, score_token_ids=None, pad_to=self.max_length
                        ),
                        data_iterator=[_SingleBatch(stream)] * len(self.models),
                        model=self.models,
                        num_microbatches=1,
                        seq_length=self.max_length,
                        micro_batch_size=len(chunk),
                        decoder_seq_length=self.max_length,
                        forward_only=True,
                    )
                if not outputs:
                    continue
                metrics = outputs[-1]
                count = float(metrics.get("rows", 0))
                total_loss += float(metrics.get("loss", 0.0)) * count
                total_correct += float(metrics.get("accuracy", 0.0)) * count
                total_rows += count

        stats = torch.tensor([total_loss, total_correct, total_rows], device=self.device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        if float(stats[2]) == 0.0:
            return {"loss": float("nan"), "accuracy": float("nan"), "rows": 0}
        return {
            "loss": float(stats[0] / stats[2]),
            "accuracy": float(stats[1] / stats[2]),
            "rows": int(stats[2]),
        }

    def flush_optimizer_to_model(self) -> None:
        """Copy fp32 master weights into the bf16 model params before saving.

        With ``use_distributed_optimizer`` the fp32 master lives in the
        optimizer's shards and the model parameters are a bf16 working copy that
        is only correct right after an optimizer step.  Saving
        ``model.state_dict()`` without flushing wrote ``max|w|=1e34`` once while
        in-loop evaluation looked healthy -- the weights on disk were the
        previous update's stale copy.
        """

        for optimizer in getattr(self.optimizer, "chained_optimizers", []) or [self.optimizer]:
            copy = getattr(optimizer, "_copy_main_params_to_model_params", None)
            if callable(copy):
                copy()

    def stage_state(self, *, dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
        """This rank's stage weights as plain CPU tensors, ready for ``torch.save``."""

        self.flush_optimizer_to_model()
        state = {}
        for name, tensor in self.model.state_dict().items():
            value = tensor.detach().to("cpu", dtype=dtype)
            if not torch.isfinite(value).all():
                raise RuntimeError(
                    f"refusing to save non-finite tensor {name!r} "
                    f"(max|w|={value.abs().max().item()})"
                )
            state[name] = value
        return state

    def destroy(self) -> None:
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()


class _SingleBatch:
    """Iterator wrapper so a one-shot batch can drive the pipeline schedule."""

    def __init__(self, inner):
        self.inner = inner

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.inner)
        except StopIteration:
            raise StopIteration from None


__all__ = ["MegatronPipelineTrainer"]
