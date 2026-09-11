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

"""Task losses for the five native retrieval model adapters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from flagscale.models.retrieval import unwrap_model


def move_features(features: Any, device: torch.device) -> Any:
    """Move nested tokenizer output to a device without losing mappings."""

    # BatchEncoding is a Mapping rather than a concrete dict. Reconstructing
    # it with a generator can silently leave tensors on CPU in Transformers 5.x.
    if isinstance(features, Mapping):
        return {key: move_features(value, device) for key, value in features.items()}
    if isinstance(features, (list, tuple)):
        return type(features)(move_features(value, device) for value in features)
    return features.to(device) if isinstance(features, Tensor) else features


def embedding_batch(model: nn.Module, inputs: list[Any], device: torch.device) -> Tensor:
    """Encode text or multimodal inputs through a registered model adapter."""

    base = unwrap_model(model)
    features = move_features(base.prepare_embedding_inputs(inputs), device)
    output = model(features)
    embeddings = output.get("sentence_embedding")
    if embeddings is None:
        embeddings = output.get("pooler_output")
    if embeddings is None:
        raise RuntimeError(f"Retrieval model output has no embedding field: {list(output)}")
    return F.normalize(embeddings.float(), p=2, dim=-1)


def embedding_loss(
    model: nn.Module,
    batch: dict[str, list[str]],
    device: torch.device,
    temperature: float,
) -> Tensor:
    """In-batch plus hard-negative contrastive loss for text encoders."""

    anchors = embedding_batch(model, batch["anchor"], device)
    positives = embedding_batch(model, batch["positive"], device)
    negatives: list[Tensor] = []
    negative_keys = sorted(key for key in batch if key.startswith("negative"))
    for key in negative_keys:
        negatives.append(embedding_batch(model, batch[key], device))

    # In-batch positives are negatives for every other query. Additional hard
    # negatives stay grouped with their originating query.
    candidates = torch.cat([positives, *negatives], dim=0)
    logits = anchors @ candidates.T / temperature
    labels = torch.arange(anchors.shape[0], device=device) * (1 + len(negatives))
    if negatives:
        grouped = torch.stack([positives, *negatives], dim=1).reshape(
            anchors.shape[0] * (1 + len(negatives)), -1
        )
        logits = anchors @ grouped.T / temperature
    return F.cross_entropy(logits, labels)


def vl_embedding_loss(
    model: nn.Module,
    batch: dict[str, list[Any]],
    device: torch.device,
    temperature: float,
) -> Tensor:
    """Symmetric image/text contrastive loss for Qwen3-VL-Embedding."""

    text_inputs = [{"text": text} for text in batch["text"]]
    image_inputs = [{"image": image} for image in batch["image"]]
    text_embeddings = embedding_batch(model, text_inputs, device)
    image_embeddings = embedding_batch(model, image_inputs, device)
    logits = text_embeddings @ image_embeddings.T / temperature
    labels = torch.arange(logits.shape[0], device=device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def reranker_batch(model: nn.Module, pairs: list[list[str]], device: torch.device) -> Tensor:
    """Score query/document pairs through the registered reranker adapter."""

    base = unwrap_model(model)
    features = move_features(base.prepare_reranker_inputs(pairs), device)
    output = model(features)
    scores = output.get("scores")
    if scores is None:
        raise RuntimeError("CrossEncoder did not produce a 'scores' field")
    return scores.squeeze(-1).float()


def reranker_loss(model: nn.Module, batch: dict[str, list[str]], device: torch.device) -> Tensor:
    """Binary positive/negative ranking loss for Qwen3-Reranker."""

    negative_keys = sorted(key for key in batch if key.startswith("negative"))
    pairs: list[list[str]] = []
    labels: list[float] = []
    for index, anchor in enumerate(batch["anchor"]):
        pairs.append([anchor, batch["positive"][index]])
        labels.append(1.0)
        for key in negative_keys:
            pairs.append([anchor, batch[key][index]])
            labels.append(0.0)
    scores = reranker_batch(model, pairs, device)
    return F.binary_cross_entropy_with_logits(scores, torch.tensor(labels, device=device))


def clip_loss(
    model: nn.Module,
    processor: Any,
    batch: dict[str, list[Any]],
    device: torch.device,
) -> Tensor:
    """Symmetric image/text CLIP loss."""

    inputs = processor(
        text=batch["text"],
        images=batch["image"],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = move_features(inputs, device)
    outputs = model(**inputs)
    labels = torch.arange(outputs.logits_per_image.shape[0], device=device)
    return 0.5 * (
        F.cross_entropy(outputs.logits_per_image, labels)
        + F.cross_entropy(outputs.logits_per_text, labels)
    )


def loss_for_batch(
    model: nn.Module,
    processor: Any,
    task: str,
    batch: dict[str, list[Any]],
    device: torch.device,
    temperature: float,
) -> Tensor:
    """Dispatch the configured task to its model-specific loss."""

    if task == "clip":
        return clip_loss(model, processor, batch, device)
    if task == "vl_embedding":
        return vl_embedding_loss(model, batch, device, temperature)
    if task == "reranker":
        return reranker_loss(model, batch, device)
    return embedding_loss(model, batch, device, temperature)


__all__ = [
    "clip_loss",
    "embedding_loss",
    "loss_for_batch",
    "move_features",
    "reranker_loss",
    "vl_embedding_loss",
]
