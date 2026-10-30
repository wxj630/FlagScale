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

"""FlagGems activation for the native retrieval trainers.

FlagGems replaces ``aten`` implementations with its own Triton kernels, which is
the entry point every later operator optimisation goes through.  The config key
``train.hardware.use_flaggems`` used to be exported as ``FLAGOS_USE_FLAGGEMS``
and then read by nothing, so setting it changed no behaviour.  This module is
what makes it real.

Usage is one call, after the device is known and before the model is built::

    from flagscale.train.utils.flaggems import enable_flaggems
    enabled = enable_flaggems(train_cfg.get("hardware", {}), device, platform)

Registering is global and irreversible for the process, so it must happen before
any kernel for the affected operators is captured (for example by CUDA graphs or
a compiled region).  Doing it right after ``init_distributed`` is safe.

Vendor selection
----------------
FlagGems ships per-vendor backends (``kunlunxin``, ``thead``/PPU, ``nvidia``,
...) and picks one from the device it finds.  ``GEMS_VENDOR`` overrides that when
a build cannot detect the device correctly -- PPU needs it because Torch-FL
presents the device through ``torch.cuda``.

Known pitfall: FlagGems defaults its Triton cache to ``~/.flaggems``.  On the
Kunlunxin host the home partition is full (and AGENTS.md forbids working there),
so the import itself fails with ``ENOSPC``.  ``FLAGGEMS_CACHE_DIR``,
``TRITON_CACHE_DIR`` and ``TORCH_EXTENSIONS_DIR`` are therefore redirected to a
writable location before importing.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

# Per-vendor overrides for builds that cannot identify the device themselves.
_VENDOR_BY_PLATFORM = {
    "flagos": "thead",
    "npu": "ascend",
    "cuda": None,   # let FlagGems detect (nvidia on A100)
    "auto": None,
}


def _redirect_caches(root: str) -> dict[str, str]:
    """Point FlagGems/Triton/Torch caches at a writable directory.

    The defaults land in ``$HOME`` which is both tiny on the vendor hosts and
    explicitly off-limits.  Returns the variables that were set, so the caller
    can log them.
    """

    os.makedirs(root, exist_ok=True)
    paths = {
        "FLAGGEMS_CACHE_DIR": os.path.join(root, "flaggems"),
        "TRITON_CACHE_DIR": os.path.join(root, "triton"),
        "TORCH_EXTENSIONS_DIR": os.path.join(root, "torch_extensions"),
    }
    for key, value in paths.items():
        os.makedirs(value, exist_ok=True)
        # Respect an explicit user setting: they may have a faster disk.
        if not os.environ.get(key):
            os.environ[key] = value
    return {k: os.environ[k] for k in paths}


def _cache_root(hardware: Any, device: Any) -> str:
    """Choose where FlagGems/JIT artefacts go.

    ``train.hardware.flaggems_cache_dir`` wins; otherwise derive something off
    the output volume so a checkpoint disk is used rather than a home dir.
    """

    explicit = None
    if hasattr(hardware, "get"):
        explicit = hardware.get("flaggems_cache_dir")
    if explicit:
        return str(explicit)

    output_dir = None
    if hasattr(hardware, "get"):
        output_dir = hardware.get("output_dir")
    # $TMPDIR is redirected to the data disk by the runner scripts on hosts whose
    # root filesystem is small.
    base = output_dir or os.environ.get("TMPDIR") or tempfile.gettempdir()
    return os.path.join(str(base), "flaggems_cache")


def enable_flaggems(hardware: Any, device: Any, platform: Any = None) -> bool:
    """Activate FlagGems if ``use_flaggems`` asks for it.

    Returns whether it was activated.  Raises only when it was requested but
    cannot be honoured -- silently training on unoptimised kernels while the
    config says otherwise is exactly the bug this replaces.  Set
    ``flaggems_required=false`` to downgrade that to a warning.
    """

    get = hardware.get if hasattr(hardware, "get") else (lambda key, default=None: default)
    requested = get("use_flaggems", False)
    if isinstance(requested, str):
        requested = requested.strip().lower() in {"1", "true", "yes", "on"}
    if not requested:
        return False

    required = get("flaggems_required", True)
    if isinstance(required, str):
        required = required.strip().lower() in {"1", "true", "yes", "on"}

    cache = _redirect_caches(_cache_root(hardware, device))
    vendor = get("flaggems_vendor") or _VENDOR_BY_PLATFORM.get(
        getattr(platform, "name", lambda: "auto")(), None
    )
    if vendor and not os.environ.get("GEMS_VENDOR"):
        os.environ["GEMS_VENDOR"] = str(vendor)

    try:
        import flag_gems
    except Exception as exc:
        message = (
            "use_flaggems=true but flag_gems could not be imported "
            f"({type(exc).__name__}: {exc}). Install it with "
            "`pip install flag_gems` in the training environment."
        )
        if required:
            raise RuntimeError(message) from exc
        print(f"[flaggems] WARNING: {message}", flush=True)
        return False

    # ``unused`` excludes ops by name.  A list, or a YAML path with an
    # ``exclude:`` key, or "default"/None for the vendor's own config.
    # OmegaConf hands lists back as ListConfig, which FlagGems does not accept as
    # a sequence -- it falls through to "load the vendor YAML" and the exclusion
    # is silently ignored.
    exclude = get("flaggems_exclude", None)
    if exclude is not None and not isinstance(exclude, str):
        exclude = [str(name) for name in exclude] or None
    if isinstance(exclude, str) and exclude.strip().lower() in {"", "none"}:
        exclude = None
    record = bool(get("flaggems_record", False))
    log_path = get("flaggems_log_path", None)

    try:
        flag_gems.enable(unused=exclude, record=record, once=True, path=log_path)
    except Exception as exc:
        message = f"flag_gems.enable(...) failed: {type(exc).__name__}: {exc}"
        if required:
            raise RuntimeError(message) from exc
        print(f"[flaggems] WARNING: {message}", flush=True)
        return False

    extra = ", ".join(f"{k}={v}" for k, v in cache.items())
    print(
        f"[flaggems] enabled vendor={os.environ.get('GEMS_VENDOR', 'auto-detect')} "
        f"version={getattr(flag_gems, '__version__', '?')} "
        f"exclude={'default' if exclude is None else exclude} ({extra})",
        flush=True,
    )
    return True


__all__ = ["enable_flaggems"]
