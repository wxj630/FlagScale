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

from pathlib import Path

import torch
from omegaconf import DictConfig

from .base import RetrievalModel
from .transformer_encoder import torch_dtype_from_config
from .registry import register_retrieval_model

# The checkpoint is trained to answer yes/no through Qwen's chat format; its
# published usage builds that framing from an explicit prefix/suffix pair and
# scores the final position.  Scoring with a generic "Query:/Passage:" prompt
# does not reproduce the pretrained behaviour, so the canonical framing is kept
# here and only the instruction is configurable.
DEFAULT_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)


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
        instruction: str = DEFAULT_INSTRUCTION,
    ):
        super().__init__(backbone)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.use_score_head = use_score_head
        self.instruction = instruction
        hidden_size = int(backbone.config.hidden_size)
        dtype = next(backbone.parameters()).dtype
        self.score_head = (
            torch.nn.Linear(hidden_size, 1, dtype=dtype) if use_score_head else None
        )
        self._prefix_ids = self.tokenizer(
            "<|im_start|>system\n" + RERANK_SYSTEM + "<|im_end|>\n<|im_start|>user\n",
            add_special_tokens=False,
        )["input_ids"]
        self._suffix_ids = self.tokenizer(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )["input_ids"]

    def prepare_reranker_inputs(self, pairs):
        bodies = [
            "<Instruct>: {i}\n<Query>: {q}\n<Document>: {d}".format(
                i=self.instruction, q=query, d=document
            )
            for query, document in pairs
        ]
        budget = max(16, self.max_length - len(self._prefix_ids) - len(self._suffix_ids))
        encoded = self.tokenizer(
            bodies,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=budget,
        )["input_ids"]
        ids = [self._prefix_ids + row + self._suffix_ids for row in encoded]
        return self.tokenizer.pad({"input_ids": ids}, padding=True, return_tensors="pt")

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
        if self.score_head is not None:
            # The scalar head is a task head outside the HF checkpoint; persist
            # it explicitly so a fine-tuned reranker reloads faithfully.
            torch.save(
                self.score_head.state_dict(),
                str(Path(output_dir) / "score_head.pt"),
            )

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
            str(model_cfg.get("instruction", DEFAULT_INSTRUCTION)),
        )
        head_path = Path(str(model_cfg.model_path)) / "score_head.pt"
        if model.score_head is not None and head_path.is_file():
            state = torch.load(str(head_path), map_location="cpu")
            model.score_head.load_state_dict(state)
        model.load_to_device(device)
        if bool(model_cfg.get("freeze_backbone", False)):
            model.freeze_backbone()
        if bool(model_cfg.get("gradient_checkpointing", False)):
            model.enable_gradient_checkpointing()
        return model
