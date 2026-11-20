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
from flagscale.models.retrieval.bge_m3 import ENSEMBLE_SPARSE_WEIGHT, SPARSE_WEIGHT


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


def bge_m3_three_way_loss(
    model: nn.Module,
    batch: dict[str, list[Any]],
    device: torch.device,
) -> Tensor:
    """Stage 3: dense + sparse + multi-vector with self-knowledge distillation.

    Ported from FlagOpen/FlagEmbedding ``research/BGE_M3/modeling.py``. The
    batch supplies one query and a group of candidates per row, so a row's
    positive sits at ``row * group_size`` in the flattened candidate list --
    that is what the ``labels * group_size`` below reproduces.

    Structure (paper section 3.2):

      * every head is trained with an InfoNCE loss against the same labels;
      * the *ensemble* score (dense + 0.3*sparse + colbert) is itself a loss
        term, which is what teaches the three heads to agree;
      * after ``self_distill_start_step`` the detached ensemble becomes a soft
        teacher for each head (self-knowledge distillation).

    ``teacher_scores`` in the batch switches to soft-label distillation from an
    external teacher instead, matching the reference's other branch.
    """

    base = unwrap_model(model)
    if not getattr(base, "unified_finetuning", False):
        raise RuntimeError(
            "bge_m3_three_way_loss needs model.unified_finetuning=true; the "
            "sparse and colbert heads do not exist otherwise"
        )

    group_size = len(batch["positive"]) // len(batch["anchor"])
    queries = _encode_bge_m3(base, batch["anchor"], device)
    passages = _encode_bge_m3(
        base,
        [t for key in _candidate_keys(batch) for t in batch[key]],
        device,
    )

    dense = base.dense_score(queries["dense"], passages["dense"])
    sparse = base.sparse_score(queries["sparse"], passages["sparse"])
    colbert = base.colbert_score(
        queries["colbert"],
        passages["colbert"],
        queries["mask"],
    )
    ensemble = dense + ENSEMBLE_SPARSE_WEIGHT * sparse + colbert

    labels = torch.arange(dense.size(0), device=device) * group_size

    teacher_scores = batch.get("teacher_scores")
    if teacher_scores is not None:
        teacher = F.softmax(torch.as_tensor(teacher_scores, device=device), dim=-1)
        loss = (
            _distill(teacher, dense, group_size)
            + _distill(teacher, sparse, group_size)
            + _distill(teacher, colbert, group_size)
            + _distill(teacher, ensemble, group_size)
        ) / 4
        return loss

    loss = (
        F.cross_entropy(dense, labels)
        + F.cross_entropy(ensemble, labels)
        + SPARSE_WEIGHT * F.cross_entropy(sparse, labels)
        + F.cross_entropy(colbert, labels)
    ) / 4

    # Self-distillation: the ensemble becomes a frozen teacher once the student
    # has had time to separate the three views. `.detach()` is what makes it
    # self-distillation rather than a fourth training signal.
    start = int(getattr(base, "self_distill_start_step", -1) or -1)
    step = int(getattr(base, "global_step", 0) or 0)
    if bool(getattr(base, "use_self_distill", False)) and start >= 0 and step > start:
        teacher = F.softmax(ensemble.detach(), dim=-1)
        distill = (
            _soft_ce(dense, teacher)
            + SPARSE_WEIGHT * _soft_ce(sparse, teacher)
            + _soft_ce(colbert, teacher)
        ) / 3
        loss = (loss + distill) / 2
    return loss


def _candidate_keys(batch: Mapping[str, Any]) -> list[str]:
    """Candidate columns in a stable order: positives first, then negatives."""

    keys = ["positive"]
    keys.extend(sorted(k for k in batch if k.startswith("negative")))
    return keys


def _encode_bge_m3(base: nn.Module, texts: list[str], device: torch.device) -> dict[str, Tensor]:
    features = move_features(base.prepare_embedding_inputs(texts), device)
    out = base(features)
    return {
        "dense": out["sentence_embedding"],
        "sparse": out["sparse_embedding"],
        "colbert": out["colbert_embedding"],
        "mask": features["attention_mask"],
    }


def _distill(teacher_targets: Tensor, student_scores: Tensor, group_size: int) -> Tensor:
    """Soft cross-entropy against a teacher, masking one candidate per group.

    Reproduces the reference's loop: the teacher's mass on each in-group
    candidate weights that candidate's loss, and the candidate is masked out of
    the denominator so each query is scored against its own group only.
    """

    labels = torch.arange(student_scores.size(0), device=student_scores.device)
    labels = labels * group_size
    loss = 0.0
    mask = torch.zeros_like(student_scores)
    for i in range(group_size):
        target = labels + i
        masked_scores = student_scores + mask
        step_loss = F.cross_entropy(masked_scores, target, reduction="none")
        loss = loss + torch.mean(teacher_targets[:, i] * step_loss)
        mask = torch.scatter(
            mask,
            dim=-1,
            index=target.unsqueeze(-1),
            value=torch.finfo(student_scores.dtype).min,
        )
    return loss


def _soft_ce(student_scores: Tensor, teacher_probs: Tensor) -> Tensor:
    log_probs = F.log_softmax(student_scores, dim=-1)
    return -torch.mean(torch.sum(log_probs * teacher_probs, dim=-1))


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
    # BGE-M3 stage 3 has three heads and a distillation term; the dense-only
    # embedding_loss above cannot express it. Selected by the model's own
    # flag so a stage-2 run (heads absent) keeps using embedding_loss.
    if getattr(unwrap_model(model), "unified_finetuning", False):
        return bge_m3_three_way_loss(model, batch, device)
    return embedding_loss(model, batch, device, temperature)


__all__ = [
    "bge_m3_three_way_loss",
    "clip_loss",
    "embedding_loss",
    "loss_for_batch",
    "move_features",
    "reranker_loss",
    "vl_embedding_loss",
]
