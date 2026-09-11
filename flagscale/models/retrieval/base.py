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

"""Common adapter contract for the native retrieval model modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying model when native DDP is active."""

    return model.module if isinstance(model, DDP) else model


class RetrievalModel(nn.Module, ABC):
    """Thin FlagScale adapter around a local pretrained model.

    The five retrieval models use different upstream libraries, but the native
    training loop only needs a small common contract: a normal ``forward``
    method, task-specific input preparation, and ``save_pretrained``.  The
    adapter deliberately does not reimplement any upstream architecture.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.backbone(*args, **kwargs)

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_cfg: Any, device: torch.device
    ) -> "RetrievalModel":
        """Load the adapter's upstream model from a local checkpoint."""

    @abstractmethod
    def prepare_embedding_inputs(self, inputs: list[Any]) -> Any:
        """Tokenize text or preprocess multimodal embedding inputs."""

    def prepare_reranker_inputs(self, pairs: list[list[str]]) -> Any:
        raise TypeError(f"{type(self).__name__} does not support reranker inputs")

    def save_pretrained(self, output_dir: str) -> None:
        self.backbone.save_pretrained(output_dir)

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def enable_gradient_checkpointing(self) -> bool:
        """Trade compute for memory by checkpointing backbone activations.

        Full fine-tuning of the 8B adapters does not fit on a single 80GB card
        once optimizer state is included, so callers enable activation
        checkpointing to shrink the activation footprint.
        """

        enable = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if enable is None:
            return False
        try:
            enable()
        except (TypeError, ValueError):
            return False
        # Cached decoder state is incompatible with checkpointed activations.
        if hasattr(self.backbone, "config"):
            self.backbone.config.use_cache = False
        return True

    def load_to_device(self, device: torch.device) -> None:
        self.to(device)
