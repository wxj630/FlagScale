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

"""CLIP ViT-L/14 native retrieval adapter."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from .base import RetrievalModel
from .registry import register_retrieval_model
from .transformer_encoder import torch_dtype_from_config


@register_retrieval_model("clip_vit_large")
class CLIPViTLargeModel(RetrievalModel):
    """Load a local Transformers CLIP ViT-L/14 checkpoint and processor."""

    def __init__(self, backbone: torch.nn.Module, processor: Any) -> None:
        super().__init__(backbone)
        self.processor = processor

    def prepare_embedding_inputs(self, inputs: list[Any]) -> Any:
        raise TypeError("CLIP uses its processor directly in the CLIP loss")

    @classmethod
    def from_pretrained(
        cls, model_cfg: DictConfig, device: torch.device
    ) -> "CLIPViTLargeModel":
        from transformers import CLIPModel, CLIPProcessor

        model_path = str(model_cfg.model_path)
        backbone = CLIPModel.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_from_config(model_cfg),
        )
        processor = CLIPProcessor.from_pretrained(model_path)
        model = cls(backbone, processor)
        model.load_to_device(device)
        if bool(model_cfg.get("freeze_backbone", False)):
            model.freeze_backbone()
        return model
