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

"""Qwen3-VL-Embedding native retrieval adapter."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from .base import RetrievalModel
from .transformer_encoder import last_token_pool, torch_dtype_from_config
from .registry import register_retrieval_model


@register_retrieval_model("qwen3_vl_embedding")
class Qwen3VLEmbeddingModel(RetrievalModel):
    """Qwen3-VL model with explicit processor and last-token pooling."""

    def __init__(self, backbone, processor, max_length: int, projection_dim: int = 0) -> None:
        super().__init__(backbone)
        self.processor = processor
        self.max_length = max_length
        hidden_size = int(backbone.config.text_config.hidden_size)
        dtype = next(backbone.parameters()).dtype
        self.projection = (
            torch.nn.Linear(hidden_size, projection_dim, bias=False, dtype=dtype)
            if projection_dim and projection_dim != hidden_size
            else torch.nn.Identity()
        )

    def prepare_embedding_inputs(self, inputs):
        conversations = []
        images = []
        has_images = any(item.get("image") is not None for item in inputs)
        for item in inputs:
            content = []
            if item.get("image") is not None:
                content.append({"type": "image", "image": item["image"]})
                images.append(item["image"])
            if item.get("text"):
                content.append({"type": "text", "text": item["text"]})
            if not content:
                content.append({"type": "text", "text": "NULL"})
            conversations.append([{"role": "user", "content": content}])
        texts = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        kwargs = {
            "text": texts,
            "padding": True,
            "truncation": True,
            "max_length": self.max_length,
            "return_tensors": "pt",
        }
        if has_images:
            kwargs["images"] = images
        return self.processor(**kwargs)

    def forward(self, features, **kwargs):
        # Transformers' Qwen3-VL RoPE helper builds a tensor from a list of
        # scalar tensors.  Torch-FL correctly keeps those scalars on ``flagos``
        # but (like several vendor backends) does not implement ``len`` for a
        # zero-dimensional tensor during that factory fallback.  Normalize the
        # scalar list to Python numbers only for this narrow upstream call;
        # model parameters and all actual activations remain on ``flagos``.
        original_tensor = torch.tensor

        def safe_tensor(data, *args, **tensor_kwargs):
            if isinstance(data, list) and data and all(
                isinstance(item, torch.Tensor) and item.ndim == 0 for item in data
            ):
                data = [item.item() for item in data]
            return original_tensor(data, *args, **tensor_kwargs)

        torch.tensor = safe_tensor
        try:
            outputs = self.backbone(
                **features,
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )
        finally:
            torch.tensor = original_tensor
        hidden = outputs.hidden_states[-1]
        embedding = self.projection(last_token_pool(hidden, features["attention_mask"]))
        return {"sentence_embedding": embedding}

    def save_pretrained(self, output_dir: str) -> None:
        self.backbone.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
        cls, model_cfg: DictConfig, device: torch.device
    ) -> "Qwen3VLEmbeddingModel":
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVisionModel
        except ImportError:  # Transformers 4.x compatibility
            from transformers import AutoModelForVision2Seq as AutoVisionModel

        processor = AutoProcessor.from_pretrained(
            str(model_cfg.model_path), local_files_only=True
        )
        conditional_model = AutoVisionModel.from_pretrained(
            str(model_cfg.model_path),
            local_files_only=True,
            torch_dtype=torch_dtype_from_config(model_cfg),
        )
        # The embedding checkpoint stores the base-model parameters under the
        # ``model.`` prefix.  Loading the conditional wrapper maps that prefix
        # correctly; retaining only its base model then drops the unused,
        # randomly initialized LM head.
        backbone = conditional_model.model
        model = cls(
            backbone,
            processor,
            int(model_cfg.get("max_length", 8192)),
            int(model_cfg.get("projection_dim", 0)),
        )
        del conditional_model
        model.load_to_device(device)
        if bool(model_cfg.get("freeze_backbone", False)):
            model.freeze_backbone()
        elif bool(model_cfg.get("freeze_vision", False)):
            # Keep the pretrained vision tower fixed and fine-tune the language
            # tower plus the projection, which is the usual single-card setup
            # for a large vision-language encoder.
            model.freeze_vision()
        return model
