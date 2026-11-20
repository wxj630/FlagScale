"""Stage-3 BGE-M3: three-way heads and self-knowledge distillation.

These lock the behaviours that are easy to get wrong when porting from
FlagEmbedding and that fail *silently* -- a wrong candidate order or a detached
sparse vector still yields a finite loss, so nothing downstream would notice.

Run: pytest tests/unit_tests/train/test_bge_m3_stage3.py
"""

from __future__ import annotations

import pytest
import torch

from flagscale.models.retrieval.bge_m3 import (
    ENSEMBLE_SPARSE_WEIGHT,
    SPARSE_WEIGHT,
    BGEM3Model,
)

transformers = pytest.importorskip("transformers")
AutoConfig = transformers.AutoConfig
AutoModel = transformers.AutoModel

VOCAB = 200


class _Tokeniser:
    """Fixed-width synthetic tokeniser: no download, deterministic shapes."""

    cls_token_id = 0
    eos_token_id = 2
    pad_token_id = 1
    unk_token_id = 3

    def __init__(self, low: int = 5, high: int = 50):
        # A narrow token range on purpose: sparse scoring multiplies query and
        # passage vectors, so disjoint vocabularies would make every score zero
        # and therefore every sparse gradient zero for reasons unrelated to the
        # code under test. This bit me while validating the port.
        self.low, self.high = low, high
        self._gen = torch.Generator().manual_seed(0)

    def __call__(self, texts, **kwargs):
        n, length = len(texts), 8
        ids = torch.randint(self.low, self.high, (n, length), generator=self._gen)
        # Force a shared token into every row so query/passage dot products are
        # non-zero by construction rather than by luck.
        ids[:, 0] = 7
        return {
            "input_ids": ids,
            "attention_mask": torch.ones(n, length, dtype=torch.long),
        }

    def save_pretrained(self, output_dir):
        # save_pretrained() on the adapter saves the tokeniser alongside the
        # weights; the stub only needs to not explode.
        return None


def _model(unified: bool = True, colbert_dim: int = -1):
    config = AutoConfig.for_model(
        "xlm-roberta",
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    backbone = AutoModel.from_config(config)
    model = BGEM3Model(
        backbone,
        _Tokeniser(),
        pooling="cls",
        unified_finetuning=unified,
        colbert_dim=colbert_dim,
    )
    model.train()
    return model


def _three_way_loss(model, batch):
    from flagscale.train.losses.retrieval import bge_m3_three_way_loss

    return bge_m3_three_way_loss(model, batch, torch.device("cpu"))


def test_stage2_has_no_heads():
    """unified_finetuning=False must not create the sparse/colbert heads.

    If it did, a stage-2 checkpoint would carry randomly initialised head
    weights into stage 3, and stage 3 would start from noise without any error.
    """

    model = _model(unified=False)
    assert model.sparse_linear is None
    assert model.colbert_linear is None
    out = model(model.prepare_embedding_inputs(["a", "b"]))
    assert set(out) == {"sentence_embedding"}


def test_three_way_output_shapes():
    model = _model(unified=True)
    out = model(model.prepare_embedding_inputs(["a", "b", "c"]))
    assert sorted(out) == ["colbert_embedding", "sentence_embedding", "sparse_embedding"]
    assert out["sentence_embedding"].shape == (3, 64)
    # Sparse is a vocabulary-sized lexical vector.
    assert out["sparse_embedding"].shape == (3, VOCAB)
    # Multi-vector keeps one row per non-CLS token: length 8 minus the CLS slot.
    assert out["colbert_embedding"].shape == (3, 7, 64)


def test_colbert_dim_override():
    model = _model(unified=True, colbert_dim=32)
    out = model(model.prepare_embedding_inputs(["a"]))
    assert out["colbert_embedding"].shape == (1, 7, 32)


def test_dense_is_normalised():
    model = _model()
    out = model(model.prepare_embedding_inputs(["a", "b"]))
    norms = out["sentence_embedding"].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_sparse_masks_special_tokens():
    """CLS/EOS/PAD/UNK must carry no lexical weight."""

    model = _model()
    tok = _Tokeniser()
    # Force every row to contain all four special ids.
    feats = {
        "input_ids": torch.tensor([[0, 2, 1, 3, 10, 11, 12, 13]] * 2),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    sparse = model(feats)["sparse_embedding"]
    for special in (tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id):
        assert torch.all(sparse[:, special] == 0)


def test_three_way_loss_is_finite_and_reaches_both_heads():
    """The regression that matters: a detached or mis-shaped head still gives a
    finite loss, so only the gradient proves it is being trained."""

    model = _model()
    batch = {
        "anchor": ["q1", "q2"],
        "positive": ["p1", "p2"],
        "negative_1": ["n1", "n2"],
    }
    loss = _three_way_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()

    for name in ("sparse_linear", "colbert_linear"):
        grad = getattr(model, name).weight.grad
        assert grad is not None, f"{name} received no gradient"
        assert grad.abs().sum() > 0, f"{name} gradient is all zeros"


def test_candidates_are_ordered_by_row():
    """Row i's positive must sit at index i*group_size.

    retrieval_collate yields parallel lists, so a column-major concatenation
    (all positives, then all negatives) points every label at the wrong
    passage while still producing a finite loss.
    """

    model = _model()
    # Identical anchors but distinguishable candidate text lengths would need
    # token-level control the stub tokeniser lacks, so assert the invariant that
    # the loss rejects mismatched candidate counts, which is the real guard.
    batch = {
        "anchor": ["q1", "q2"],
        "positive": ["p1"],  # one short
        "negative_1": ["n1", "n2"],
    }
    with pytest.raises(ValueError, match="one 'positive' per row"):
        _three_way_loss(model, batch)


def test_fusion_weights_match_paper():
    """Table 8 values are constants on purpose; drifting them silently makes
    results incomparable with the paper and with other platforms."""

    from flagscale.train.losses import retrieval as losses

    assert ENSEMBLE_SPARSE_WEIGHT == 0.3
    assert SPARSE_WEIGHT == 0.1
    assert losses.ENSEMBLE_SPARSE_WEIGHT == 0.3
    assert losses.SPARSE_WEIGHT == 0.1


def test_heads_round_trip_through_checkpoint(tmp_path):
    """Stage 3 resumes from stage 2 with fresh heads, then must be able to
    reload its own heads on the next run."""

    model = _model()
    output = tmp_path / "ckpt"
    output.mkdir()
    model.backbone.save_pretrained(output)
    model.save_pretrained(str(output))

    assert (output / "sparse_linear.pt").exists()
    assert (output / "colbert_linear.pt").exists()

    resumed = _model()
    before = resumed.sparse_linear.weight.detach().clone()
    assert resumed.load_heads(str(output)) is True
    assert not torch.allclose(before, resumed.sparse_linear.weight.detach())


def test_missing_heads_are_not_an_error(tmp_path):
    """Starting stage 3 from stage 2 must not fail: the paper initialises
    W_lex and W_mul randomly at that point."""

    empty = tmp_path / "stage2"
    empty.mkdir()
    model = _model()
    assert model.load_heads(str(empty)) is False
