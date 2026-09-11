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

"""Registry and factory for native retrieval model adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from .base import RetrievalModel

RETRIEVAL_MODEL_REGISTRY: dict[str, type[RetrievalModel]] = {}


def register_retrieval_model(name: str) -> Callable[[type[RetrievalModel]], type[RetrievalModel]]:
    """Register a model adapter under its Hydra ``model.type`` name."""

    def decorator(cls: type[RetrievalModel]) -> type[RetrievalModel]:
        if name in RETRIEVAL_MODEL_REGISTRY and RETRIEVAL_MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"Retrieval model '{name}' is already registered")
        RETRIEVAL_MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def build_retrieval_model(
    name: str,
    model_cfg: DictConfig,
    device: torch.device,
) -> tuple[RetrievalModel, Any]:
    """Instantiate a registered retrieval adapter from local model assets."""

    if name not in RETRIEVAL_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown retrieval model '{name}'. "
            f"Available: {sorted(RETRIEVAL_MODEL_REGISTRY)}"
        )
    model = RETRIEVAL_MODEL_REGISTRY[name].from_pretrained(model_cfg, device)
    return model, getattr(model, "processor", None)

