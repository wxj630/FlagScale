# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Dependency-light Transformer encoder helpers for native retrieval models.

The adapters in this package are ordinary ``torch.nn.Module`` objects.  They
use the Transformers model/tokenizer APIs already present in FlagScale's
training environments and implement pooling/checkpoint handling locally;
no separate embedding framework is required on this path.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from torch import Tensor

from .base import RetrievalModel


def torch_dtype_from_config(model_cfg: DictConfig) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(str(model_cfg.get("torch_dtype", "bf16")).lower(), torch.bfloat16)


def last_token_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    positions = attention_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, positions]


def mean_pool(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    weights = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class TransformerTextEncoder(RetrievalModel):
    """A Hugging Face Transformer plus tokenizer and explicit pooling."""

    def __init__(self, backbone: torch.nn.Module, tokenizer: Any, pooling: str) -> None:
        super().__init__(backbone)
        self.tokenizer = tokenizer
        self.pooling = pooling
        self.max_length = 8192

    def prepare_embedding_inputs(self, inputs: list[str]) -> dict[str, Tensor]:
        return self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, features: dict[str, Tensor], **kwargs: Any) -> dict[str, Tensor]:
        outputs = self.backbone(**features, return_dict=True, **kwargs)
        hidden = outputs.last_hidden_state
        mask = features["attention_mask"]
        if self.pooling == "cls":
            embedding = hidden[:, 0]
        elif self.pooling == "mean":
            embedding = mean_pool(hidden, mask)
        else:
            embedding = last_token_pool(hidden, mask)
        return {"sentence_embedding": embedding}

    def save_pretrained(self, output_dir: str) -> None:
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
