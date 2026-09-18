"""Print one retrieval config field, for use from the training driver script.

Keeps the driver free of inline Python string interpolation:

    python conf_value.py examples/retrieval/conf/train/bge_m3.yaml model.model_path
"""

from __future__ import annotations

import sys

from omegaconf import OmegaConf


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: conf_value.py <config.yaml> <dotted.key>")
    cfg = OmegaConf.load(sys.argv[1])
    value = OmegaConf.select(cfg, sys.argv[2])
    if value is None:
        raise SystemExit(f"config {sys.argv[1]} has no key {sys.argv[2]!r}")
    print(value)


if __name__ == "__main__":
    main()
