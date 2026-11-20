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

"""Collator for RetroMAE: two independent masks and a per-position diagonal.

Adapted from FlagOpen/FlagEmbedding
``research/old-examples/pretrain/retromae_pretrain/data.py``.

The non-obvious part is ``decoder_attention_mask``. It is not a 2-D padding mask
but a ``[B, L, L]`` matrix: for every decode position ``i`` the model sees a
*different* mask of the source sentence, and position ``i`` itself is always
visible. That is what makes the decoder reconstruct from ``[CLS]`` instead of
copying the input -- if every position saw the same mask, the task would
degenerate into plain MLM.

Dependency note: transformers 5.x removed ``DataCollatorForWholeWordMask``
entirely, which the reference subclasses for ``_whole_word_mask`` and
``torch_mask_tokens``. Both are small, so they are reimplemented here using only
tokenizer primitives that exist in 5.x. Whole-word masking matters for BGE-M3
because XLM-R uses sentencepiece: masking half a word teaches the model to guess
subword endings from their stems, which the paper's bottleneck objective does
not want.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import torch

MIN_MASKS = 128


@dataclass
class RetroMAECollator:
    """Two-mask collator with a per-position decoder visibility matrix."""

    tokenizer: Any
    max_seq_length: int = 512
    # The paper's asymmetric rates: the encoder is masked at the same rate as
    # the decoder here, but the decoder additionally receives the encoder's
    # [CLS] and a mostly-clean copy, so the asymmetry comes from information
    # access rather than mask ratio.
    encoder_mlm_probability: float = 0.15
    decoder_mlm_probability: float = 0.15
    max_masks: int = MIN_MASKS
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def __call__(self, examples):
        input_ids_batch = []
        attention_mask_batch = []
        encoder_mlm_mask_batch = []
        decoder_labels_batch = []
        decoder_matrix_attention_mask_batch = []

        for text in examples:
            encoded = self.tokenizer.encode(
                text, max_length=self.max_seq_length, truncation=True
            )
            if len(encoded) < 3:
                # CLS + at least one token + SEP; anything shorter has nothing
                # to reconstruct.
                continue
            tokens = [self.tokenizer.convert_ids_to_tokens(tid) for tid in encoded]

            encoder_mask = self._whole_word_mask(tokens, self.encoder_mlm_probability)

            pool_size = min(len(tokens), self.max_masks)
            masks = [
                self._whole_word_mask(tokens, self.decoder_mlm_probability)
                for _ in range(pool_size)
            ]

            matrix_attention_mask = []
            for i in range(len(tokens)):
                idx = self._rng.randint(0, pool_size - 1)
                decoder_mask = list(masks[idx])
                # Position i must stay visible; otherwise the decoder is asked
                # to predict a token it cannot see.
                decoder_mask[i] = 1
                matrix_attention_mask.append(decoder_mask)

            input_ids_batch.append(torch.tensor(encoded))
            attention_mask_batch.append(torch.ones(len(encoded), dtype=torch.long))

            labels = list(encoded)
            labels[0] = -100  # CLS is never predicted
            labels[-1] = -100  # nor SEP
            decoder_labels_batch.append(torch.tensor(labels))
            encoder_mlm_mask_batch.append(torch.tensor(encoder_mask))
            # 1 = hidden. The reference inverts here to match transformers'
            # additive attention-mask convention.
            decoder_matrix_attention_mask_batch.append(
                1 - torch.tensor(matrix_attention_mask)
            )

        if not input_ids_batch:
            raise ValueError("every example was shorter than CLS+1+SEP")

        input_ids = _pad(input_ids_batch, self.tokenizer.pad_token_id)
        attention_mask = _pad(attention_mask_batch, 0)
        origin_input_ids = input_ids.clone()
        encoder_mask_t = _pad(encoder_mlm_mask_batch, 0)
        encoder_input_ids, encoder_labels = self._mask_tokens(input_ids, encoder_mask_t)
        decoder_labels = _pad(decoder_labels_batch, -100)
        decoder_attention = _pad(decoder_matrix_attention_mask_batch, 0)

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": attention_mask,
            "encoder_labels": encoder_labels,
            "decoder_input_ids": origin_input_ids,
            "decoder_attention_mask": decoder_attention,
            "decoder_labels": decoder_labels,
        }

    # ---- masking primitives (replacing the removed HF collator) ----------

    def _whole_word_mask(self, tokens: list[str], probability: float) -> list[int]:
        """Return a 0/1 mask that never splits a word mid-piece.

        Mirrors the removed HF implementation: a new word starts when the
        tokenizer's word-boundary marker appears (or for the first token after
        punctuation), and every subword of a word shares the decision.
        """

        cand = []
        for i, token in enumerate(tokens):
            if i == 0 or token.startswith("##") or token.startswith("\u0120"):
                cand.append([i])
            elif len(cand) and not _is_punctuation(token[0]):
                cand[-1].append(i)
            else:
                cand.append([i])

        mask = [0] * len(tokens)
        budget = max(1, int(len(tokens) * probability)) if tokens else 0
        self._rng.shuffle(cand)
        chosen = 0
        for word in cand:
            if chosen >= budget:
                break
            for idx in word:
                mask[idx] = 1
            chosen += len(word)
        return mask

    def _mask_tokens(self, input_ids: torch.Tensor, mask: torch.Tensor):
        """Apply 80/10/10 replace-random/keep/mask and build the MLM labels."""

        if self.tokenizer.mask_token_id is None:
            raise ValueError("tokenizer has no mask token; RetroMAE needs one")

        encoder_input_ids = input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = mask.to(torch.float)
        probability_matrix[input_ids == self.tokenizer.pad_token_id] = 0
        probability_matrix[input_ids == self.tokenizer.cls_token_id] = 0
        probability_matrix[input_ids == self.tokenizer.sep_token_id] = 0

        masked = probability_matrix.bool()

        # 80% -> [MASK]
        replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        encoder_input_ids[replace_mask] = self.tokenizer.mask_token_id

        # 10% -> random token
        random_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked
            & ~replace_mask
        )
        vocab = getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer)
        random_words = torch.randint(vocab, labels.shape, dtype=torch.long)
        encoder_input_ids[random_mask] = random_words[random_mask]

        # The remaining 10% keep their original id.

        labels[~masked] = -100
        return encoder_input_ids, labels


def _is_punctuation(char: str) -> bool:
    import unicodedata

    cp = ord(char)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(char).startswith("P")


def _pad(tensors: list[torch.Tensor], value) -> torch.Tensor:
    """Right-pad a list of 1-D or 2-D tensors to the batch maximum."""

    if tensors[0].dim() == 1:
        length = max(t.size(0) for t in tensors)
        out = torch.full((len(tensors), length), value, dtype=tensors[0].dtype)
        for row, t in enumerate(tensors):
            out[row, : t.size(0)] = t
        return out

    rows = max(t.size(0) for t in tensors)
    cols = max(t.size(1) for t in tensors)
    out = torch.full((len(tensors), rows, cols), value, dtype=tensors[0].dtype)
    for idx, t in enumerate(tensors):
        out[idx, : t.size(0), : t.size(1)] = t
    return out
