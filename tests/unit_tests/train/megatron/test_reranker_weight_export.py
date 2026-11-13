# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight-conversion tests for the Megatron reranker path.

The round-trip ``hf -> stage -> hf`` is the only thing standing between a
33-hour pipeline-parallel run and any evaluable artifact: the Megatron trainer
writes per-stage files, and ``evaluate_retrieval.py`` only reads HF directories.
When the export mapping is wrong the failure is silent -- the model loads, scores
run, and the ranking degrades to chance -- so these tests pin the two properties
that make the numbers trustworthy: the split partitions the layers, and the
export reconstructs the original tensors exactly.

Only the pure mapping functions are exercised.  Building a real ``GPTModel``
needs Megatron and a GPU, which is why the mapping was factored out of the model
construction in the first place.
"""

import pytest
import torch

pytest.importorskip("torch")

from flagscale.models.retrieval.qwen3_reranker_megatron import (  # noqa: E402
    _merge_stages,
    interleave_qkv,
    megatron_to_hf_state,
    split_qkv,
)


def _tiny_hf(num_layers: int = 6, hidden: int = 8, heads: int = 4, groups: int = 2) -> dict:
    """Config shaped like Qwen3, small enough to build in a few microseconds."""

    return {
        "num_hidden_layers": num_layers,
        "num_attention_heads": heads,
        "num_key_value_heads": groups,
        "head_dim": hidden // heads,
        "hidden_size": hidden,
    }


def _fake_megatron_stage(hf: dict, pp_size: int, pp_rank: int, seed: int) -> dict:
    """A stage dict with the key layout ``megatron_to_hf_state`` expects.

    Values are deterministic per (layer, role) so the merge test can assert that
    each global layer came from the stage that owns it.
    """

    torch.manual_seed(seed)
    per_stage = hf["num_hidden_layers"] // pp_size
    hidden = hf["hidden_size"]
    head_dim = hf["head_dim"]
    groups = hf["num_key_value_heads"]
    q_rows = hf["num_attention_heads"] * head_dim

    stage = {}
    if pp_rank == 0:
        stage["embedding.word_embeddings.weight"] = torch.arange(hidden, dtype=torch.float32)
    for local in range(per_stage):
        prefix = f"decoder.layers.{local}"
        stage[f"{prefix}.input_layernorm.weight"] = torch.full((hidden,), float(local))
        stage[f"{prefix}.self_attention.linear_qkv.weight"] = torch.zeros(
            q_rows + 2 * groups * head_dim, hidden
        )
        stage[f"{prefix}.self_attention.q_layernorm.weight"] = torch.zeros(head_dim)
        stage[f"{prefix}.self_attention.k_layernorm.weight"] = torch.zeros(head_dim)
        stage[f"{prefix}.self_attention.linear_proj.weight"] = torch.zeros(hidden, hidden)
        stage[f"{prefix}.pre_mlp_layernorm.weight"] = torch.zeros(hidden)
        stage[f"{prefix}.mlp.linear_fc1.weight"] = torch.zeros(2 * hidden, hidden)
        stage[f"{prefix}.mlp.linear_fc2.weight"] = torch.zeros(hidden, hidden)
    if pp_rank == pp_size - 1:
        stage["decoder.final_layernorm.weight"] = torch.ones(hidden)
        stage["output_layer.weight"] = torch.zeros(16, hidden)
    return stage


@pytest.mark.parametrize("pp_size", [1, 2, 3, 6])
def test_merge_partitions_all_layers_exactly_once(pp_size: int) -> None:
    """Every layer appears once: no gap (unloadable model) and no duplicate."""

    hf = _tiny_hf(num_layers=6)
    stages = [_fake_megatron_stage(hf, pp_size, r, seed=r) for r in range(pp_size)]
    merged = _merge_stages(stages, hf, pp_size)

    layer_ids = {
        int(name.split(".")[2])
        for name in merged
        if name.startswith("model.layers.")
    }
    assert layer_ids == set(range(6)), f"pp={pp_size} covered {sorted(layer_ids)}"
    assert "model.embed_tokens.weight" in merged
    assert "model.norm.weight" in merged
    assert "lm_head.weight" in merged


def test_export_rejects_pp_that_does_not_divide_the_layers() -> None:
    """A non-divisible split must raise instead of dropping the remainder.

    ``megatron_to_hf_state`` uses integer division for the per-stage layer count,
    so 5 layers over 2 stages would export 2 layers per stage and silently lose
    layer 4.  The merged model then loads with ``strict=False`` and scores
    nonsense -- the exact failure that is hardest to notice, because every
    downstream number stays finite.
    """

    hf = _tiny_hf(num_layers=5)
    stages = [_fake_megatron_stage(hf, 2, r, seed=r) for r in range(2)]
    with pytest.raises(RuntimeError, match="not divisible by"):
        _merge_stages(stages, hf, 2)


def test_export_rejects_a_stage_missing_a_layer() -> None:
    """A truncated stage (e.g. a partially written file) must not export cleanly.

    Either guard is acceptable: the key lookup fails first when the whole layer is
    gone (``KeyError``), and the layer-count check catches a stage that exports
    fewer layers than the config declares (``RuntimeError``).  What matters is that
    neither path returns a short state_dict that loads with ``strict=False``.
    """

    hf = _tiny_hf(num_layers=2)
    stages = [_fake_megatron_stage(hf, 2, r, seed=r) for r in range(2)]
    for name in [k for k in stages[1] if k.startswith("decoder.layers.")]:
        stages[1].pop(name)
    with pytest.raises((RuntimeError, KeyError)):
        _merge_stages(stages, hf, 2)


@pytest.mark.parametrize("groups", [1, 2, 4])
def test_qkv_interleave_round_trips(groups: int) -> None:
    """``split_qkv`` must invert ``interleave_qkv`` for every grouping.

    The QKV layout is the mapping most likely to be wrong in a way that still
    produces finite scores -- a wrong interleave shifts them silently.
    """

    heads, head_dim, hidden = 8, 4, 32
    q = torch.randn(heads * head_dim, hidden)
    k = torch.randn(groups * head_dim, hidden)
    v = torch.randn(groups * head_dim, hidden)

    fused = interleave_qkv(q, k, v, num_query_groups=groups)
    q2, k2, v2 = split_qkv(
        fused,
        num_attention_heads=heads,
        num_query_groups=groups,
        head_dim=head_dim,
    )
    assert torch.equal(q, q2)
    assert torch.equal(k, k2)
    assert torch.equal(v, v2)


def test_export_preserves_values_for_a_random_stage() -> None:
    """Values survive stage -> HF for a single stage, including the embeddings.

    This is the regression guard for the fp32/bf16 and view-aliasing bugs: the
    exporter casts and clones before writing, so the comparison is on values, not
    on dtypes.
    """

    hf = _tiny_hf(num_layers=1)
    stage = _fake_megatron_stage(hf, 1, 0, seed=7)
    exported = megatron_to_hf_state(stage, hf, pp_size=1, pp_rank=0)

    assert torch.equal(
        exported["model.embed_tokens.weight"], stage["embedding.word_embeddings.weight"]
    )
    assert torch.equal(exported["model.layers.0.input_layernorm.weight"], torch.zeros(8))
    assert exported["model.layers.0.self_attn.q_proj.weight"].shape == (8, 8)
    assert exported["model.layers.0.self_attn.k_proj.weight"].shape == (4, 8)
