# Copyright (c) 2026, BAAI. All rights reserved.

PLATFORMS = {}


def register_platforms() -> None:
    """Register all available platforms."""

    # Torch-FL must be imported by the training entrypoint before this module
    # is reached.  Its import registers ``torch.flagos`` and the corresponding
    # distributed backend; this adapter only exposes those APIs to FlagScale.
    from .platform_flagos import PlatformFlagOS

    platform_flagos = PlatformFlagOS()
    if platform_flagos.is_available():
        PLATFORMS["flagos"] = platform_flagos

    from .platform_cuda import PlatformCUDA

    platform_cuda = PlatformCUDA()
    if platform_cuda.is_available():
        PLATFORMS["cuda"] = platform_cuda

    from .platform_npu import PlatformNPU

    platform_npu = PlatformNPU()
    if platform_npu.is_available():
        PLATFORMS["npu"] = platform_npu

    from .platform_musa import PlatformMUSA

    platform_musa = PlatformMUSA()
    if platform_musa.is_available():
        PLATFORMS["musa"] = platform_musa

    from .platform_txda import PlatformTXDA

    platform_txda = PlatformTXDA()
    if platform_txda.is_available():
        PLATFORMS["txda"] = platform_txda
