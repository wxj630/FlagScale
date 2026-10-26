"""Check that the retrieval seeding actually makes a run repeatable.

Two things get verified, because they fail for different reasons:

1. ``seed_everything`` covers numpy and the device RNG, not just python's.
2. The train loader yields the same batch order twice for the same seed, and a
   different order for a different seed -- with ``num_workers > 0``, since that
   is where worker RNG usually leaks in.

Run it on any host that can import FlagScale:

    python -m tools.seed_repro_test
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

from omegaconf import OmegaConf

from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.train_retrieval import seed_everything


def _order(loader, count: int = 4) -> list[list[str]]:
    """The first ``count`` batches' anchors, as a comparable value."""

    batches = []
    for batch in itertools.islice(loader, count):
        batches.append(list(batch["anchor"]))
    return batches


def check_seeds_cover_numpy_and_device() -> None:
    seed_everything(1234, rank=0, device=torch.device("cpu"), deterministic=False)
    first = (np.random.rand(), torch.rand(1).item(), torch.initial_seed())
    seed_everything(1234, rank=0, device=torch.device("cpu"), deterministic=False)
    second = (np.random.rand(), torch.rand(1).item(), torch.initial_seed())
    assert first == second, f"same seed diverged: {first} vs {second}"
    # rank must change the stream, otherwise every replica draws identically.
    seed_everything(1234, rank=1, device=torch.device("cpu"), deterministic=False)
    third = (np.random.rand(), torch.rand(1).item(), torch.initial_seed())
    assert third != first, "rank was not folded into the seed"
    print(f"OK seeds: numpy+torch+device reproducible, rank differs ({first[2]} vs {third[2]})")


def check_loader_repeatable(data_path: str, seed: int, workers: int) -> None:
    cfg = OmegaConf.create(
        {
            "path": data_path,
            "train_path": data_path,
            "validation_path": data_path,
            "test_path": data_path,
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "split_modulo": 20,
            "shuffle_buffer": 4096,
            "length_sort_window": 0,
            "num_negatives": 1,
            "num_workers": workers,
        }
    )

    def run(seed_value: int) -> list[list[str]]:
        seed_everything(seed_value, rank=0, device=torch.device("cpu"), deterministic=False)
        train_loader, _, _ = build_split_loaders(
            "reranker", cfg, batch_size=2, pin_memory=False, seed=seed_value
        )
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(0)
        return _order(train_loader)

    first = run(seed)
    second = run(seed)
    other = run(seed + 1)

    assert first == second, "same seed produced a different batch order"
    assert first != other, "a different seed produced the same order -- seed not applied"
    print(
        f"OK loader (workers={workers}): same seed -> identical order, "
        f"seed+1 -> different order ({len(first)} batches compared)"
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    check_seeds_cover_numpy_and_device()
    check_loader_repeatable(args.data, args.seed, args.workers)
    check_loader_repeatable(args.data, args.seed, 0)
    print("ALL SEED CHECKS PASSED")


if __name__ == "__main__":
    main()
