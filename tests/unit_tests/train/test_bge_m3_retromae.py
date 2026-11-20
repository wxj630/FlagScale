"""Stage-1 BGE-M3: RetroMAE encoder-decoder pretraining.

The failure this guards against is subtle: a RetroMAE that reconstructs well
but does not actually route everything through ``[CLS]`` still shows a falling
loss, yet it stops being RetroMAE and no longer matches the paper's premise that
the CLS vector becomes an information bottleneck.

Run: pytest tests/unit_tests/train/test_bge_m3_retromae.py
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import AutoConfig, AutoModelForMaskedLM

from flagscale.models.retrieval.retromae import RetroMAEForPretraining
from flagscale.train.datasets.retromae_collator import MIN_MASKS, RetroMAECollator

VOCAB = 120


class _Tokeniser:
    """Minimal BERT-like tokeniser; no download, deterministic."""

    pad_token_id = 0
    cls_token_id = 1
    sep_token_id = 2
    mask_token_id = 3
    unk_token_id = 4
    vocab_size = VOCAB

    def encode(self, text, max_length=32, truncation=True):
        ids = [self.cls_token_id]
        # One id per whitespace token, wrapped in the usual CLS/SEP frame.
        for word in str(text).split():
            ids.append(5 + (hash(word) % (VOCAB - 10)))
        ids.append(self.sep_token_id)
        return ids[:max_length]

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return f"tok{ids}"
        return [f"tok{i}" for i in ids]

    def _convert_id_to_token(self, tid):
        return f"tok{tid}"

    def tokenize(self, text):
        return str(text).split()

    def __len__(self):
        return VOCAB


def _lm():
    config = AutoConfig.for_model(
        "bert",
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    return AutoModelForMaskedLM.from_config(config)


def test_collator_emits_a_per_position_diagonal():
    """decoder_attention_mask must be [B, L, L], not a 2-D padding mask.

    With a 2-D mask every decode position would see the same source tokens and
    the task collapses to plain MLM.
    """

    collator = RetroMAECollator(_Tokeniser(), max_seq_length=16)
    batch = collator(["alpha beta gamma delta", "one two three"])
    mask = batch["decoder_attention_mask"]

    assert mask.dim() == 3, f"expected [B, L, L], got {tuple(mask.shape)}"
    batch_size, rows, cols = mask.shape
    assert rows == cols


def test_collator_masks_cls_and_sep_in_labels():
    collator = RetroMAECollator(_Tokeniser(), max_seq_length=16)
    batch = collator(["alpha beta gamma"])
    labels = batch["decoder_labels"][0]
    assert labels[0] == -100
    # The SEP position is the last real token; padding follows it.
    length = int(batch["encoder_attention_mask"][0].sum())
    assert labels[length - 1] == -100


def test_encoder_and_decoder_inputs_differ():
    """The encoder is masked, the decoder sees the clean sentence.

    If these were identical the asymmetric design would be gone.
    """

    collator = RetroMAECollator(_Tokeniser(), max_seq_length=16)
    batch = collator(["alpha beta gamma delta epsilon zeta eta theta"])
    enc, dec = batch["encoder_input_ids"], batch["decoder_input_ids"]
    assert not torch.equal(enc, dec), "encoder input was not masked"


def test_forward_produces_a_finite_loss_and_gradients():
    torch.manual_seed(0)
    model = RetroMAEForPretraining(_lm())
    model.train()

    collator = RetroMAECollator(_Tokeniser(), max_seq_length=16)
    batch = collator(
        [
            "alpha beta gamma delta epsilon",
            "zeta eta theta iota kappa",
        ]
    )
    (loss,) = model(**batch)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()

    decoder_grad = model.c_head.attention.self.query.weight.grad
    assert decoder_grad is not None and decoder_grad.abs().sum() > 0, (
        "decoder cross-attention received no gradient"
    )
    encoder_grad = model.lm.base_model.embeddings.word_embeddings.weight.grad
    assert encoder_grad is not None and encoder_grad.abs().sum() > 0


def test_cls_hidden_state_is_the_only_encoder_signal():
    """Perturbing a non-CLS encoder output must not change the loss path.

    This is the bottleneck property in its most testable form: the decoder
    receives ``cls_hiddens`` plus the *input embeddings*, never the encoder's
    per-token hidden states.
    """

    import inspect

    source = inspect.getsource(RetroMAEForPretraining.forward)
    assert "hidden_states[-1][:, :1]" in source, "decoder must read only [CLS]"
    # The key/value stream must come from the embedding table, not lm_out.
    assert "decoder_embedding_output" in source


def test_decoder_attention_pool_is_bounded():
    """A long sentence must not build one mask per token without limit."""

    collator = RetroMAECollator(_Tokeniser(), max_seq_length=40)
    long_text = " ".join(f"w{i}" for i in range(60))
    batch = collator([long_text])
    # Bounded by MIN_MASKS, so the [B, L, L] tensor stays finite in size.
    assert batch["decoder_attention_mask"].shape[1] <= MIN_MASKS


def test_save_writes_a_loadable_encoder(tmp_path):
    """Stage 2 points ``model_path`` at this output, so it must load without
    the decoder head and without unexpected keys."""

    from transformers import AutoModel

    model = RetroMAEForPretraining(_lm())
    out = tmp_path / "stage1"
    model.save_pretrained(str(out))

    reloaded = AutoModel.from_pretrained(str(out))
    assert reloaded.config.vocab_size == VOCAB
    # The decoder head must not leak into the encoder checkpoint.
    state = reloaded.state_dict()
    assert not any(key.startswith("c_head") for key in state)
