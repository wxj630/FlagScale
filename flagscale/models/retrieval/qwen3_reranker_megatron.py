"""Qwen3-Reranker as a Megatron ``GPTModel`` (pipeline-parallel training).

This is the Megatron counterpart of :mod:`qwen3_reranker`, which wraps the HF
``transformers`` model.  Both score a (query, document) pair the same way --
``logit[last, yes] - logit[last, no]`` at the final prompt position -- so the
prompt framing and the score-token ids come from the HF adapter rather than
being duplicated here.  Only the model construction and the HF<->Megatron weight
mapping are Megatron-specific.

Why a separate module instead of importing ``GPTModel`` in ``qwen3_reranker``:
the HF adapter is what DDP/FSDP2 use and it must not require Megatron to be
installed.  Keeping the Megatron build behind its own module means a CUDA/PPU
host with no Megatron-LM-FL can still import the retrieval package.

Everything here is TP=1.  Qwen3-8B has 32 attention heads, so a 3-card box can
only split it by pipeline stage; a tensor-parallel shard of the group-interleaved
QKV layout has no validated mapping, and shipping an unvalidated one is worse
than refusing it.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .qwen3_reranker import DEFAULT_INSTRUCTION, RERANK_SYSTEM


def prompt_prefix() -> str:
    """Chat framing before the instruction, shared with the HF adapter."""
    return "<|im_start|>system\n" + RERANK_SYSTEM + "<|im_end|>\n<|im_start|>user\n"


def prompt_suffix() -> str:
    """Assistant turn opener after the body, shared with the HF adapter."""
    return "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_prompt_body(instruction: str, query: str, document: str) -> str:
    return "<Instruct>: {i}\n<Query>: {q}\n<Document>: {d}".format(
        i=instruction, q=query, d=document
    )


def read_hf_config(model_path: str | Path) -> dict:
    return json.loads((Path(model_path) / "config.json").read_text())


def build_megatron_config(
    hf: dict,
    *,
    pipeline_model_parallel_size: int = 1,
    tensor_model_parallel_size: int = 1,
    micro_batch_size: int = 1,
    recompute: bool = True,
    attention_backend: str = "flash",
    dtype: torch.dtype = torch.bfloat16,
):
    """Translate the HF config into the knob set Megatron needs for Qwen3."""

    from megatron.core.transformer.enums import AttnBackend
    from megatron.core.transformer.transformer_config import TransformerConfig

    if tensor_model_parallel_size != 1:
        raise NotImplementedError(
            "tensor_model_parallel_size > 1 is not implemented: the Qwen3 GQA "
            "QKV layout is group-interleaved and a TP shard of it has not been "
            "validated against the HF checkpoint."
        )
    return TransformerConfig(
        num_layers=hf["num_hidden_layers"],
        hidden_size=hf["hidden_size"],
        num_attention_heads=hf["num_attention_heads"],
        num_query_groups=hf["num_key_value_heads"],
        kv_channels=hf["head_dim"],
        ffn_hidden_size=hf["intermediate_size"],
        # Qwen3's MLP is SwiGLU: two linears in fc1, silu applied to the gate half.
        activation_func=F.silu,
        gated_linear_unit=True,
        normalization="RMSNorm",
        layernorm_epsilon=hf["rms_norm_eps"],
        add_bias_linear=hf.get("attention_bias", False),
        qk_layernorm=True,
        bf16=dtype == torch.bfloat16,
        params_dtype=dtype,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        # Megatron requires the pipeline dtype at construction time when
        # pp_size > 1; assigning it afterwards raises ValueError.
        pipeline_dtype=dtype if pipeline_model_parallel_size > 1 else None,
        sequence_parallel=False,
        # The vendored local path has no fused kernels for these on XPU; every
        # prior P800 run disabled them, and the unfused paths are the ones with
        # measured numerics.
        bias_activation_fusion=False,
        bias_dropout_fusion=False,
        masked_softmax_fusion=False,
        persist_layer_norm=False,
        apply_rope_fusion=False,
        gradient_accumulation_fusion=False,
        attention_backend=AttnBackend[attention_backend],
        attention_softmax_in_fp32=True,
        recompute_granularity="full" if recompute else None,
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=1 if recompute else None,
    )


def build_megatron_model(
    config,
    hf: dict,
    *,
    pre_process: bool = True,
    post_process: bool = True,
    seq_length: int = 640,
):
    """One pipeline stage of the Qwen3 reranker."""

    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    spec = get_gpt_layer_local_spec(
        num_experts=None, qk_layernorm=True, normalization="RMSNorm"
    )
    # The checkpoint pads the vocabulary to 151669 and does not tie its
    # embedding to lm_head, so the output layer is a real parameter.
    return GPTModel(
        config=config,
        transformer_layer_spec=spec,
        vocab_size=hf["vocab_size"],
        max_sequence_length=hf.get("max_position_embeddings", 40960),
        pre_process=pre_process,
        post_process=post_process,
        position_embedding_type="rope",
        rotary_percent=1.0,
        rotary_base=int(hf.get("rope_theta", 10000)),
        share_embeddings_and_output_weights=False,
    )


def interleave_qkv(
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    *,
    num_query_groups: int,
) -> torch.Tensor:
    """Build Megatron's fused QKV weight from the three HF projections.

    ``linear_qkv`` is reshaped to ``[..., num_query_groups, (q_per_group + 2) *
    head_dim]`` and split along the last axis, so per group the rows are that
    group's query heads, then its one key head, then its one value head -- not
    all queries followed by all keys and values.  Getting this wrong shifts the
    scores by more than 0.5.
    """

    head_dim = k_proj.shape[0] // num_query_groups
    q_per_group = q_proj.shape[0] // num_query_groups // head_dim
    groups = []
    for group in range(num_query_groups):
        q_start = group * q_per_group * head_dim
        q_end = q_start + q_per_group * head_dim
        k_start = group * head_dim
        k_end = k_start + head_dim
        groups.append(q_proj[q_start:q_end])
        groups.append(k_proj[k_start:k_end])
        groups.append(v_proj[k_start:k_end])
    return torch.cat(groups, dim=0)


def split_qkv(
    qkv: torch.Tensor,
    *,
    num_attention_heads: int,
    num_query_groups: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_qkv`; used by the HF exporter."""

    q_per_group = num_attention_heads // num_query_groups
    q_parts, k_parts, v_parts = [], [], []
    cursor = 0
    for _ in range(num_query_groups):
        q_rows = q_per_group * head_dim
        q_parts.append(qkv[cursor : cursor + q_rows])
        cursor += q_rows
        k_parts.append(qkv[cursor : cursor + head_dim])
        cursor += head_dim
        v_parts.append(qkv[cursor : cursor + head_dim])
        cursor += head_dim
    return torch.cat(q_parts, 0), torch.cat(k_parts, 0), torch.cat(v_parts, 0)


def hf_to_megatron_state(
    hf_state: dict[str, torch.Tensor], hf: dict, *, pp_size: int, pp_rank: int
) -> dict[str, torch.Tensor]:
    """Map one pipeline stage's slice of the HF checkpoint to Megatron names."""

    num_layers = hf["num_hidden_layers"]
    assert num_layers % pp_size == 0, (
        f"{num_layers} layers do not split evenly over {pp_size} pipeline stages"
    )
    per_stage = num_layers // pp_size
    first = pp_rank * per_stage
    last = first + per_stage

    out: dict[str, torch.Tensor] = {}
    if pp_rank == 0:
        out["embedding.word_embeddings.weight"] = hf_state["model.embed_tokens.weight"]

    for global_layer in range(first, last):
        src = f"model.layers.{global_layer}"
        dst = f"decoder.layers.{global_layer - first}"
        out[f"{dst}.input_layernorm.weight"] = hf_state[f"{src}.input_layernorm.weight"]
        out[f"{dst}.self_attention.linear_qkv.weight"] = interleave_qkv(
            hf_state[f"{src}.self_attn.q_proj.weight"],
            hf_state[f"{src}.self_attn.k_proj.weight"],
            hf_state[f"{src}.self_attn.v_proj.weight"],
            num_query_groups=hf["num_key_value_heads"],
        )
        out[f"{dst}.self_attention.q_layernorm.weight"] = hf_state[f"{src}.self_attn.q_norm.weight"]
        out[f"{dst}.self_attention.k_layernorm.weight"] = hf_state[f"{src}.self_attn.k_norm.weight"]
        out[f"{dst}.self_attention.linear_proj.weight"] = hf_state[f"{src}.self_attn.o_proj.weight"]
        out[f"{dst}.pre_mlp_layernorm.weight"] = hf_state[f"{src}.post_attention_layernorm.weight"]
        out[f"{dst}.mlp.linear_fc1.weight"] = torch.cat(
            [hf_state[f"{src}.mlp.gate_proj.weight"], hf_state[f"{src}.mlp.up_proj.weight"]],
            dim=0,
        )
        out[f"{dst}.mlp.linear_fc2.weight"] = hf_state[f"{src}.mlp.down_proj.weight"]

    if pp_rank == pp_size - 1:
        out["decoder.final_layernorm.weight"] = hf_state["model.norm.weight"]
        out["output_layer.weight"] = hf_state["lm_head.weight"]
    return out


def megatron_to_hf_state(
    megatron_state: dict[str, torch.Tensor], hf: dict, *, pp_size: int, pp_rank: int
) -> dict[str, torch.Tensor]:
    """Inverse mapping, so a trained stage can be evaluated as HF weights."""

    num_layers = hf["num_hidden_layers"]
    per_stage = num_layers // pp_size
    first = pp_rank * per_stage

    out: dict[str, torch.Tensor] = {}
    if pp_rank == 0:
        out["model.embed_tokens.weight"] = megatron_state["embedding.word_embeddings.weight"]

    for global_layer in range(first, first + per_stage):
        dst = f"model.layers.{global_layer}"
        src = f"decoder.layers.{global_layer - first}"
        out[f"{dst}.input_layernorm.weight"] = megatron_state[f"{src}.input_layernorm.weight"]
        q, k, v = split_qkv(
            megatron_state[f"{src}.self_attention.linear_qkv.weight"],
            num_attention_heads=hf["num_attention_heads"],
            num_query_groups=hf["num_key_value_heads"],
            head_dim=hf["head_dim"],
        )
        out[f"{dst}.self_attn.q_proj.weight"] = q
        out[f"{dst}.self_attn.k_proj.weight"] = k
        out[f"{dst}.self_attn.v_proj.weight"] = v
        out[f"{dst}.self_attn.q_norm.weight"] = megatron_state[f"{src}.self_attention.q_layernorm.weight"]
        out[f"{dst}.self_attn.k_norm.weight"] = megatron_state[f"{src}.self_attention.k_layernorm.weight"]
        out[f"{dst}.self_attn.o_proj.weight"] = megatron_state[f"{src}.self_attention.linear_proj.weight"]
        out[f"{dst}.post_attention_layernorm.weight"] = megatron_state[f"{src}.pre_mlp_layernorm.weight"]
        gate, up = torch.chunk(megatron_state[f"{src}.mlp.linear_fc1.weight"], 2, dim=0)
        out[f"{dst}.mlp.gate_proj.weight"] = gate
        out[f"{dst}.mlp.up_proj.weight"] = up
        out[f"{dst}.mlp.down_proj.weight"] = megatron_state[f"{src}.mlp.linear_fc2.weight"]

    if pp_rank == pp_size - 1:
        out["model.norm.weight"] = megatron_state["decoder.final_layernorm.weight"]
        out["lm_head.weight"] = megatron_state["output_layer.weight"]
    return out


def load_stage_weights(model, stage_file: Path, pp_rank: int) -> int:
    """Load one stage's converted weights; returns the tensor count."""

    if not stage_file.is_file():
        raise FileNotFoundError(
            f"missing {stage_file}; run "
            "`python -m flagscale.models.retrieval.qwen3_reranker_megatron convert`"
        )
    weights = torch.load(stage_file, map_location="cpu")
    own = {k: v for k, v in model.state_dict().items() if k in weights}
    missing = set(weights) - set(own)
    if missing:
        raise RuntimeError(
            f"stage{pp_rank}: converter produced unknown tensors: {sorted(missing)[:5]}"
        )
    model.load_state_dict({**model.state_dict(), **weights}, strict=False)
    return len(weights)


__all__ = [
    "DEFAULT_INSTRUCTION",
    "RERANK_SYSTEM",
    "build_megatron_config",
    "build_megatron_model",
    "build_prompt_body",
    "hf_to_megatron_state",
    "interleave_qkv",
    "load_stage_weights",
    "megatron_to_hf_state",
    "prompt_prefix",
    "prompt_suffix",
    "read_hf_config",
    "split_qkv",
]


def _convert_main(argv: list[str] | None = None) -> None:
    """``python -m flagscale.models.retrieval.qwen3_reranker_megatron convert``.

    Run once, on CPU, before the first pipeline-parallel run: each training rank
    then loads only its own ``stage{K}.pt``, which keeps the loader free of
    distributed collectives and lets the files be inspected with ``torch.load``.
    """

    import argparse

    from safetensors.torch import load_file

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF checkpoint directory")
    parser.add_argument("--pp", type=int, default=1, help="pipeline model parallel size")
    parser.add_argument("--out", required=True, help="output directory for stage{K}.pt")
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    hf = read_hf_config(model_path)

    index = model_path / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(
            {model_path / name for name in json.loads(index.read_text())["weight_map"].values()}
        )
    else:
        single = model_path / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"no safetensors weights under {model_path}")
        shards = [single]

    hf_state: dict[str, torch.Tensor] = {}
    for shard in shards:
        hf_state.update(load_file(str(shard)))
    print(f"loaded {len(hf_state)} HF tensors from {model_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pp_rank in range(args.pp):
        stage = hf_to_megatron_state(hf_state, hf, pp_size=args.pp, pp_rank=pp_rank)
        stage = {name: tensor.contiguous() for name, tensor in stage.items()}
        destination = out_dir / f"stage{pp_rank}.pt"
        torch.save(stage, destination)
        total = sum(t.numel() for t in stage.values())
        print(f"stage{pp_rank}: {len(stage)} tensors, {total / 1e9:.2f}B params -> {destination}")


if __name__ == "__main__":
    _convert_main()
