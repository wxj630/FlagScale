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

"""BGE-M3 native retrieval adapter: dense, sparse and multi-vector heads.

Ported from FlagOpen/FlagEmbedding ``research/BGE_M3/modeling.py``, the
reference implementation of the paper's stages 2 and 3. Reproducing the
three-way scoring and the self-knowledge-distillation loss here is what lets a
FlagScale run match the paper instead of approximating it with dense-only
contrastive training.

Faithful to the reference:

  * ``sparse_linear`` maps each token's hidden state to one lexical weight;
    ``colbert_linear`` projects each token to a ``colbert_dim`` vector. Both are
    randomly initialised, which the paper also does at the start of stage 3
    (``W_lex`` / ``W_mul`` are new parameters there).
  * The fusion weights 0.3 and 0.1 are hard-coded rather than configurable.
    Paper Table 8 uses these exact values, and exposing them as knobs would
    silently produce numbers that cannot be compared with the paper or with
    other platforms.
  * ``sparse_embedding`` keeps the reference's train/eval split. The training
    path builds a ``[batch, seq, vocab]`` tensor and max-pools over the
    sequence; eval uses ``scatter_reduce``. The split exists because the
    training path's in-place scatter breaks autograd when reused for inference
    (upstream issue #1364). Merging them into one branch changes the loss.

Stage coverage: ``unified_finetuning=False`` is stage 2 (contrastive
pretraining, dense only) -- the two heads are not even created, so a stage-2
checkpoint cannot carry randomly-initialised head weights into stage 3. Set
``unified_finetuning=True`` for stage 3. Stage 1 (RetroMAE) is a different
model structure and lives in its own trainer, not here.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor, nn

from .registry import register_retrieval_model
from .transformer_encoder import TransformerTextEncoder, torch_dtype_from_config

logger = logging.getLogger(__name__)

# Paper Table 8 / reference modeling.py. Deliberately not configurable.
SPARSE_WEIGHT = 0.1
ENSEMBLE_SPARSE_WEIGHT = 0.3


@register_retrieval_model("bge_m3")
class BGEM3Model(TransformerTextEncoder):
    """XLM-R encoder with dense, sparse and multi-vector outputs."""

    @classmethod
    def from_pretrained(cls, model_cfg: DictConfig, device: torch.device) -> "BGEM3Model":
        from transformers import AutoModel, AutoTokenizer

        model_path = str(model_cfg.model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        backbone = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch_dtype_from_config(model_cfg),
        )
        unified = bool(model_cfg.get("unified_finetuning", False))
        model = cls(
            backbone,
            tokenizer,
            pooling="cls",
            unified_finetuning=unified,
            colbert_dim=int(model_cfg.get("colbert_dim", -1) or -1),
            temperature=float(model_cfg.get("temperature", 1.0) or 1.0),
        )
        model.max_length = int(model_cfg.get("max_length", 8192))
        # Resuming stage 3: the heads are stored beside the HF weights as
        # colbert_linear.pt / sparse_linear.pt, matching the reference layout.
        # Missing heads are the normal stage-2 -> stage-3 handoff.
        if unified:
            model.load_heads(model_path)
        model.load_to_device(device)
        if bool(model_cfg.get("freeze_backbone", False)):
            model.freeze_backbone()
        return model

    def __init__(
        self,
        backbone: nn.Module,
        tokenizer,
        pooling: str = "cls",
        unified_finetuning: bool = False,
        colbert_dim: int = -1,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(backbone, tokenizer, pooling)
        hidden = int(getattr(backbone.config, "hidden_size"))
        self.unified_finetuning = bool(unified_finetuning)
        self.colbert_dim = colbert_dim
        self.temperature = float(temperature)
        self.vocab_size = int(getattr(backbone.config, "vocab_size"))

        if self.unified_finetuning:
            self.sparse_linear = nn.Linear(hidden, 1)
            self.colbert_linear = nn.Linear(
                hidden, hidden if colbert_dim == -1 else colbert_dim
            )
            logger.info(
                "initialised sparse_linear and colbert_linear from scratch "
                "(expected at the start of stage 3, not during inference)"
            )
        else:
            self.sparse_linear = None
            self.colbert_linear = None

    # ---- heads -----------------------------------------------------------

    def sparse_embedding(
        self, hidden_state: Tensor, input_ids: Tensor, return_embedding: bool = True
    ) -> Tensor:
        """Lexical weights scattered into vocabulary space."""

        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding:
            return token_weights

        if self.training:
            sparse = torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                self.vocab_size,
                dtype=token_weights.dtype,
                device=token_weights.device,
            )
            sparse = torch.scatter(
                sparse, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights
            )
            sparse = torch.max(sparse, dim=1).values
        else:
            sparse = torch.zeros(
                input_ids.size(0),
                self.vocab_size,
                dtype=token_weights.dtype,
                device=token_weights.device,
            )
            sparse = sparse.scatter_reduce(
                dim=-1, index=input_ids, src=token_weights.squeeze(-1), reduce="amax"
            )

        # Never let a special token carry lexical weight. Use getattr because
        # tokenizers differ: XLM-R (BGE-M3) has eos_token_id but some BERT
        # variants do not, and the reference implementation would raise there.
        unused = [
            getattr(self.tokenizer, "cls_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "sep_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "unk_token_id", None),
        ]
        unused = [t for t in unused if t is not None and t >= 0]
        if unused:
            sparse[:, unused] *= 0.0
        return sparse

    def colbert_embedding(self, last_hidden_state: Tensor, mask: Tensor) -> Tensor:
        """One vector per token, skipping position 0 (the CLS slot)."""

        vecs = self.colbert_linear(last_hidden_state[:, 1:])
        return vecs * mask[:, 1:][:, :, None].float()

    def forward(self, features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        outputs = self.backbone(**features, return_dict=True, **kwargs)
        hidden = outputs.last_hidden_state
        mask = features["attention_mask"]

        if self.pooling == "cls":
            dense = hidden[:, 0]
        else:
            dense = super().forward(features)["sentence_embedding"]
        result: dict[str, Tensor] = {"sentence_embedding": F.normalize(dense.float(), dim=-1)}

        if self.unified_finetuning:
            result["sparse_embedding"] = self.sparse_embedding(hidden, features["input_ids"])
            result["colbert_embedding"] = F.normalize(
                self.colbert_embedding(hidden, mask).float(), dim=-1
            )
        return result

    # ---- scores ----------------------------------------------------------

    def dense_score(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature
        return scores.view(q_reps.size(0), -1)

    def sparse_score(self, q_reps: Tensor, p_reps: Tensor) -> Tensor:
        scores = torch.matmul(q_reps, p_reps.transpose(0, 1)) / self.temperature
        return scores.view(q_reps.size(0), -1)

    def colbert_score(self, q_reps: Tensor, p_reps: Tensor, q_mask: Tensor) -> Tensor:
        token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
        return scores / self.temperature

    # ---- persistence -----------------------------------------------------

    def save_pretrained(self, output_dir: str) -> None:
        super().save_pretrained(output_dir)
        if self.unified_finetuning:
            torch.save(self.sparse_linear.state_dict(), f"{output_dir}/sparse_linear.pt")
            torch.save(self.colbert_linear.state_dict(), f"{output_dir}/colbert_linear.pt")

    def load_heads(self, model_dir: str) -> bool:
        """Load the sparse/colbert heads if present; report whether they were.

        Absent heads are normal when stage 3 starts from stage 2's output: the
        paper initialises W_lex and W_mul randomly at that point.
        """

        import os

        sparse_path = os.path.join(model_dir, "sparse_linear.pt")
        colbert_path = os.path.join(model_dir, "colbert_linear.pt")
        if not (os.path.exists(sparse_path) and os.path.exists(colbert_path)):
            return False
        self.sparse_linear.load_state_dict(torch.load(sparse_path, map_location="cpu"))
        self.colbert_linear.load_state_dict(torch.load(colbert_path, map_location="cpu"))
        logger.info("loaded existing sparse_linear and colbert_linear from %s", model_dir)
        return True
