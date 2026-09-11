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

"""Native retrieval model adapters used by the FlagScale example."""

from .bge_m3 import BGEM3Model
from .base import RetrievalModel, unwrap_model
from .clip_vit_large import CLIPViTLargeModel
from .qwen3_embedding import Qwen3EmbeddingModel
from .qwen3_reranker import Qwen3RerankerModel
from .qwen3_vl_embedding import Qwen3VLEmbeddingModel
from .registry import (
    RETRIEVAL_MODEL_REGISTRY,
    build_retrieval_model,
    register_retrieval_model,
)

__all__ = [
    "BGEM3Model",
    "CLIPViTLargeModel",
    "Qwen3EmbeddingModel",
    "Qwen3RerankerModel",
    "Qwen3VLEmbeddingModel",
    "RetrievalModel",
    "RETRIEVAL_MODEL_REGISTRY",
    "build_retrieval_model",
    "register_retrieval_model",
    "unwrap_model",
]
