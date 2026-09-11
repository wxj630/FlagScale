# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Torch-FL platform adapter.

Torch-FL registers the ``flagos`` device through PyTorch's PrivateUse1
extension mechanism.  Keeping this adapter in FlagScale means training code
only talks to the platform interface; importing Torch-FL remains an entrypoint
concern and no model module needs to know about a vendor runtime.
"""

import torch

from .platform_base import PlatformBase


class PlatformFlagOS(PlatformBase):
    """FlagScale adapter for the Torch-FL ``flagos`` device."""

    def name(self) -> str:
        return "flagos"

    def is_available(self) -> bool:
        try:
            return bool(torch.flagos.is_available())
        except (AttributeError, RuntimeError):
            return False

    def set_device(self, device_index):
        torch.flagos.set_device(device_index)

    def device(self, device_index=None):
        return torch.device("flagos", device_index)

    def device_count(self) -> int:
        return int(torch.flagos.device_count())

    def dist_backend(self) -> str:
        # Torch-FL registers ProcessGroupFlagOS under this standard backend
        # name.  It selects FlagCX first and a vendor-native fallback second.
        return "flagos"

    def manual_seed_all(self, seed):
        torch.flagos.manual_seed_all(seed)

    def amp_device_type(self) -> str:
        return "flagos"

    def supports_pin_memory(self) -> bool:
        # CUDA-boxing Torch-FL uses the host pinned-memory path for async copies.
        return True

