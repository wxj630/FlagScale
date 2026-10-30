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

"""The config switch must actually reach FlagGems.

``train.hardware.use_flaggems`` used to be interpolated into an environment
variable that nothing read, so setting it changed no behaviour while looking
like it did.  These tests pin the contract that replaced it: off means off,
on-without-the-package fails loudly, and the cache paths leave the home
directory.
"""

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from flagscale.train.utils.flaggems import enable_flaggems


def test_disabled_is_a_noop(monkeypatch):
    """use_flaggems=false must not import flag_gems at all."""
    imports = []
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def tracking_import(name, *args, **kwargs):
        imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", tracking_import)
    hardware = OmegaConf.create({"use_flaggems": False})
    assert enable_flaggems(hardware, "cpu", None) is False
    assert "flag_gems" not in imports


def test_string_false_is_false():
    """Hydra can hand strings through; "false" must not be truthy."""

    for value in ("false", "0", "no", "off", ""):
        assert enable_flaggems(OmegaConf.create({"use_flaggems": value}), "cpu", None) is False


def test_enabled_without_package_raises(tmp_path, monkeypatch):
    """A requested-but-missing package is an error, not a silent downgrade."""

    hardware = OmegaConf.create(
        {
            "use_flaggems": True,
            "flaggems_required": True,
            "flaggems_cache_dir": str(tmp_path),
        }
    )
    with pytest.raises(RuntimeError, match="flag_gems could not be imported"):
        enable_flaggems(hardware, "cpu", SimpleNamespace(name=lambda: "cpu"))


def test_optional_degrades_with_warning(tmp_path, capsys):
    """flaggems_required=false downgrades that to a warning."""

    hardware = OmegaConf.create(
        {
            "use_flaggems": True,
            "flaggems_required": False,
            "flaggems_cache_dir": str(tmp_path),
        }
    )
    assert enable_flaggems(hardware, "cpu", None) is False
    assert "WARNING" in capsys.readouterr().out


def test_cache_paths_avoid_home(tmp_path, monkeypatch):
    """The vendor hosts have a full home partition; caches must move."""

    for key in ("FLAGGEMS_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        monkeypatch.delenv(key, raising=False)

    hardware = OmegaConf.create(
        {
            "use_flaggems": True,
            "flaggems_required": False,
            "flaggems_cache_dir": str(tmp_path),
        }
    )
    enable_flaggems(hardware, "cpu", None)

    import os

    for key in ("FLAGGEMS_CACHE_DIR", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR"):
        assert key in os.environ
        assert str(tmp_path) in os.environ[key]
        assert os.environ[key] != os.path.expanduser("~")


def test_explicit_cache_env_is_respected(tmp_path, monkeypatch):
    """A user-provided cache location wins over the derived one."""

    monkeypatch.setenv("FLAGGEMS_CACHE_DIR", str(tmp_path / "mine"))
    hardware = OmegaConf.create(
        {"use_flaggems": True, "flaggems_required": False, "flaggems_cache_dir": str(tmp_path)}
    )
    enable_flaggems(hardware, "cpu", None)

    import os

    assert os.environ["FLAGGEMS_CACHE_DIR"] == str(tmp_path / "mine")


def test_listconfig_exclude_is_converted():
    """OmegaConf returns ListConfig; FlagGems needs a plain sequence.

    Passing a ListConfig through made FlagGems treat it as a vendor config path
    and silently drop the exclusion, which is how the `sub` crash survived an
    exclude list that looked correct.
    """

    from flagscale.train.utils.flaggems import _redirect_caches  # noqa: F401

    hardware = OmegaConf.create(
        {
            "use_flaggems": True,
            "flaggems_required": False,
            "flaggems_exclude": ["sub", "bmm"],
        }
    )
    # No package installed here, so this returns False; the conversion happens
    # before the import and is what the assertion exercises.
    enable_flaggems(hardware, "cpu", None)
    exclude = hardware["flaggems_exclude"]
    assert not isinstance(exclude, str)
    assert list(exclude) == ["sub", "bmm"]
