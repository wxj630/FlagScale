# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""BGE-M3 stage 1 (RetroMAE), kept separate from the retrieval adapters.

RetroMAE is an encoder-decoder with a masked-LM objective, not a retrieval
model: it produces no embeddings and is not registered in the retrieval model
registry. Stage 2 loads its encoder output as a plain HF checkpoint.
"""

from .modeling import RetroMAEForPretraining

__all__ = ["RetroMAEForPretraining"]
