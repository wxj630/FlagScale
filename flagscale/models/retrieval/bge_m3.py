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

"""BGE-M3 native retrieval adapter."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from .transformer_encoder import TransformerTextEncoder, torch_dtype_from_config
from .registry import register_retrieval_model


@register_retrieval_model("bge_m3")
class BGEM3Model(TransformerTextEncoder):
    """XLM-R encoder with the checkpoint's CLS pooling contract."""

    @classmethod
    def from_pretrained(cls, model_cfg: DictConfig, device: torch.device) -> "BGEM3Model":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_cfg.model_path), local_files_only=True
        )
        backbone = AutoModel.from_pretrained(
            str(model_cfg.model_path),
            local_files_only=True,
            torch_dtype=torch_dtype_from_config(model_cfg),
        )
        model = cls(backbone, tokenizer, pooling="cls")
        model.max_length = int(model_cfg.get("max_length", 8192))
        model.load_to_device(device)
        if bool(model_cfg.get("freeze_backbone", False)):
            model.freeze_backbone()
        return model
