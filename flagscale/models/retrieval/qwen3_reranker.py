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

"""Qwen3-Reranker native retrieval adapter."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from .base import RetrievalModel
from .transformer_encoder import torch_dtype_from_config
from .registry import register_retrieval_model


@register_retrieval_model("qwen3_reranker")
class Qwen3RerankerModel(RetrievalModel):
    """Qwen3 causal LM scored by its learned true/false token logits."""

    def __init__(
        self,
        backbone,
        tokenizer,
        max_length: int,
        true_token_id: int,
        false_token_id: int,
        use_score_head: bool = False,
    ):
        super().__init__(backbone)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.use_score_head = use_score_head
        hidden_size = int(backbone.config.hidden_size)
        dtype = next(backbone.parameters()).dtype
        self.score_head = (
            torch.nn.Linear(hidden_size, 1, dtype=dtype) if use_score_head else None
        )

    def prepare_reranker_inputs(self, pairs):
        prompt = "Given a web search query, retrieve relevant passages that answer the query\n"
        texts = [prompt + "Query: " + query + "\nPassage: " + document for query, document in pairs]
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def prepare_embedding_inputs(self, inputs):
        raise TypeError("Qwen3-Reranker only supports query/document pairs")

    def forward(self, features, **kwargs):
        if self.use_score_head:
            outputs = self.backbone.model(
                **features, return_dict=True, **kwargs
            )
            hidden = outputs.last_hidden_state
            positions = features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            scores = self.score_head(hidden[rows, positions]).squeeze(-1)
            return {"scores": scores}
        outputs = self.backbone(**features, return_dict=True, **kwargs)
        logits = outputs.logits
        positions = features["attention_mask"].to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
        rows = torch.arange(logits.shape[0], device=logits.device)
        final_logits = logits[rows, positions]
        scores = final_logits[:, self.true_token_id] - final_logits[:, self.false_token_id]
        return {"scores": scores}

    def save_pretrained(self, output_dir: str) -> None:
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    @classmethod
    def from_pretrained(
        cls, model_cfg: DictConfig, device: torch.device
    ) -> "Qwen3RerankerModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_cfg.model_path), local_files_only=True
        )
        backbone = AutoModelForCausalLM.from_pretrained(
            str(model_cfg.model_path),
            local_files_only=True,
            torch_dtype=torch_dtype_from_config(model_cfg),
        )
        model = cls(
            backbone,
            tokenizer,
            int(model_cfg.max_length),
            int(model_cfg.get("true_token_id", 9693)),
            int(model_cfg.get("false_token_id", 2152)),
            bool(model_cfg.get("use_score_head", False)),
        )
        model.load_to_device(device)
        if bool(model_cfg.get("freeze_backbone", False)):
            model.freeze_backbone()
        return model
