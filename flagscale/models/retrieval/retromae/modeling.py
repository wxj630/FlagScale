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

"""BGE-M3 stage 1: RetroMAE pretraining.

Ported from FlagOpen/FlagEmbedding
``research/old-examples/pretrain/retromae_pretrain/modeling.py``.

What the stage does, and why it is not "just MLM":

  * An asymmetric encoder-decoder. The encoder sees a heavily masked sentence
    (default 15%) and its task is to compress it into ``[CLS]``. A separate,
    randomly-initialised decoder must then reconstruct the *whole* sentence
    from ``[CLS]`` plus a lightly masked copy of the input.
  * Because the decoder gets only ``[CLS]`` to summarise the encoder side,
    ``[CLS]`` is forced to become an information bottleneck -- and ``[CLS]`` is
    exactly what dense retrieval scores with. That objective/usage match is the
    reason this stage helps a dense retriever at all, and it is why a plain MLM
    head would not be an acceptable substitute.
  * The loss is the sum of the encoder's MLM loss and the decoder's
    reconstruction loss, matching the reference.

Two deviations from the reference, both deliberate:

  * The reference hard-codes XLM-R/BERT attribute probing (``self.lm.bert``,
    ``self.lm.roberta``, ``self.lm.cls``, ``self.lm.lm_head``). We resolve the
    embedding table and the LM head through a small helper that also handles
    XLM-R's ``lm_head`` living on the wrapper rather than on the backbone,
    because BGE-M3 is XLM-R and the reference's fallback branch would otherwise
    be reached for a different architecture.
  * ``save_pretrained`` writes a plain HF model the later stages can load, not
    the reference's ``encoder_model/`` subdirectory plus a full ``state_dict``
    pickle. Stage 2 must be able to point ``model_path`` at the result, and the
    pickle duplicates the weights on disk.
"""

from __future__ import annotations

import logging
import os

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import MaskedLMOutput

from .enhanced_decoder import BertLayerForDecoder

logger = logging.getLogger(__name__)


class RetroMAEForPretraining(nn.Module):
    """Encoder-decoder with a ``[CLS]`` bottleneck.

    ``lm`` must be a masked-LM model (``AutoModelForMaskedLM``), because both
    losses are cross-entropy over the vocabulary.
    """

    def __init__(self, lm: nn.Module, decoder_mlm_probability: float = 0.15) -> None:
        super().__init__()
        self.lm = lm
        self.decoder_embeddings = _embedding_table(lm)
        self.c_head = BertLayerForDecoder(lm.config)
        self.c_head.apply(lm._init_weights)
        self.cross_entropy = nn.CrossEntropyLoss()
        self.decoder_mlm_probability = decoder_mlm_probability

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.lm.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        encoder_input_ids: Tensor,
        encoder_attention_mask: Tensor,
        encoder_labels: Tensor,
        decoder_input_ids: Tensor,
        decoder_attention_mask: Tensor,
        decoder_labels: Tensor,
    ) -> tuple[Tensor]:
        lm_out: MaskedLMOutput = self.lm(
            encoder_input_ids,
            encoder_attention_mask,
            labels=encoder_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        # The bottleneck: everything the decoder learns about the source
        # sentence arrives through this single vector.
        cls_hiddens = lm_out.hidden_states[-1][:, :1]

        decoder_embedding_output = self.decoder_embeddings(input_ids=decoder_input_ids)
        hiddens = torch.cat([cls_hiddens, decoder_embedding_output[:, 1:]], dim=1)

        # Cross-attention query: positional embeddings with [CLS] broadcast into
        # every position. Key/value: the lightly-masked input plus [CLS].
        cls_hiddens = cls_hiddens.expand(hiddens.size(0), hiddens.size(1), hiddens.size(2))
        query = self.decoder_embeddings(inputs_embeds=cls_hiddens)

        matrix_attention_mask = _extended_attention_mask(
            self.lm, decoder_attention_mask
        )

        hiddens = self.c_head(
            query=query,
            key=hiddens,
            value=hiddens,
            attention_mask=matrix_attention_mask,
        )[0]
        _, decoder_loss = self.mlm_loss(hiddens, decoder_labels)

        return (decoder_loss + lm_out.loss,)

    def mlm_loss(self, hiddens: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        pred_scores = _lm_head(self.lm)(hiddens)
        loss = self.cross_entropy(
            pred_scores.view(-1, self.lm.config.vocab_size),
            labels.view(-1),
        )
        return pred_scores, loss

    def save_pretrained(self, output_dir: str) -> None:
        """Write the encoder as a plain HF model for stage 2 to consume.

        The decoder head is not part of the retrieval model, so it is dropped
        here; keeping it would make stage 2's ``from_pretrained`` fail on
        unexpected keys.
        """

        os.makedirs(output_dir, exist_ok=True)
        backbone = _backbone(self.lm)
        backbone.save_pretrained(output_dir)
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> "RetroMAEForPretraining":
        from transformers import AutoModelForMaskedLM

        local = {"local_files_only": True}
        local.update(kwargs.pop("hf_kwargs", {}))
        lm = AutoModelForMaskedLM.from_pretrained(model_path, **local, **kwargs)
        return cls(lm)


def _extended_attention_mask(lm: nn.Module, attention_mask: Tensor) -> Tensor:
    """Turn a 0/1 visibility mask into an additive mask, or -inf where masked.

    transformers 5.x removed ``ModuleUtilsMixin.get_extended_attention_mask``,
    which the reference calls. The transformation is small and architecture
    independent, so it is reimplemented rather than shimmed. The 4.x method is
    still preferred when present so behaviour on older hosts is unchanged.
    """

    legacy = getattr(lm, "get_extended_attention_mask", None)
    if callable(legacy):
        return legacy(attention_mask, attention_mask.shape, attention_mask.device)

    extended = attention_mask.to(dtype=torch.float32)
    if extended.dim() == 2:
        # [B, L] -> [B, 1, 1, L] so it broadcasts over heads and query positions.
        extended = extended[:, None, None, :]
    extended = (1.0 - extended) * torch.finfo(torch.float32).min
    return extended


def _backbone(lm: nn.Module) -> nn.Module:
    """Return the encoder without the MLM head.

    BGE-M3 is XLM-R, so the order matters: ``roberta`` (XLM-R) before ``bert``.
    """

    for name in ("roberta", "bert", "model"):
        candidate = getattr(lm, name, None)
        if candidate is not None:
            return candidate
    raise AttributeError(
        f"cannot find an encoder on {type(lm).__name__}; expected .roberta/.bert/.model"
    )

    """Return the encoder without the MLM head.

    BGE-M3 is XLM-R, so the order matters: ``roberta`` (XLM-R) before ``bert``.
    """

    for name in ("roberta", "bert", "model"):
        candidate = getattr(lm, name, None)
        if candidate is not None:
            return candidate
    raise AttributeError(
        f"cannot find an encoder on {type(lm).__name__}; expected .roberta/.bert/.model"
    )


def _embedding_table(lm: nn.Module) -> nn.Module:
    return _backbone(lm).embeddings


def _lm_head(lm: nn.Module) -> nn.Module:
    # XLM-R exposes lm_head on the wrapper; BERT exposes cls on the backbone.
    for name in ("lm_head", "cls"):
        head = getattr(lm, name, None)
        if head is not None:
            return head
    raise AttributeError(f"{type(lm).__name__} has neither .lm_head nor .cls")
