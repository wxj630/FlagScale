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

"""Streaming datasets used by the native retrieval training example.

The native training entrypoint should contain the model and optimization
logic, while dataset I/O stays in :mod:`flagscale.train.datasets`.  The
datasets in this module deliberately implement the PyTorch
``IterableDataset`` interface: T2Ranking is too large to index in memory and
the COCO shards are already packaged as WebDataset-style tar files.
"""

from __future__ import annotations

import io
import os
import random
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def split_accepts(index: int, split: str, modulo: int = 20) -> bool:
    """Return whether a deterministic row belongs to ``split``.

    T2Ranking ships only train shards.  For the native retrieval example we
    reserve 90%/5%/5% of rows for train/validation/test without materializing
    another copy of the dataset.  COCO's separately supplied test directory
    uses the ``all`` split and therefore bypasses this partition.
    """

    split = str(split).lower()
    if split in {"all", "none"}:
        return True
    modulo = max(3, int(modulo))
    bucket = index % modulo
    if split in {"train", "training"}:
        return bucket < modulo - 2
    if split in {"validation", "valid", "val"}:
        return bucket == modulo - 2
    if split in {"test", "testing"}:
        return bucket == modulo - 1
    raise ValueError(f"Unsupported dataset split: {split}")


def _distributed_context() -> tuple[int, int]:
    """Read torchrun's rank variables without requiring an initialized group."""

    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _shard_context() -> tuple[int, int]:
    """Combine torchrun rank and DataLoader worker id into one shard.

    An ``IterableDataset`` is materialized independently in every worker, so
    without worker-aware sharding each of the ``num_workers`` processes would
    yield the full stream and duplicate samples.  Pairing rank with worker id
    gives every (rank, worker) a distinct slice of the split.
    """

    rank, world_size = _distributed_context()
    info = get_worker_info()
    if info is None:
        return rank, world_size
    return rank * info.num_workers + info.id, world_size * info.num_workers


def _buffered_shuffle(stream: Iterator[Any], buffer_size: int, seed: int) -> Iterator[Any]:
    """Shuffle a streaming dataset with a bounded reservoir buffer.

    T2Ranking stores all rows of one query next to each other.  Feeding that
    order straight into the contrastive loss puts several positives of the same
    anchor in one batch, where they are treated as each other's negatives.  A
    bounded buffer breaks the locality without indexing the whole split.
    """

    if buffer_size <= 0:
        yield from stream
        return
    rng = random.Random(seed)
    buffer: list[Any] = []
    for item in stream:
        if len(buffer) < buffer_size:
            buffer.append(item)
            continue
        index = rng.randrange(buffer_size)
        yield buffer[index]
        buffer[index] = item
    rng.shuffle(buffer)
    yield from buffer


def _text_length(row: dict[str, Any]) -> int:
    """Proxy for tokenized length, available before tokenization."""

    return len(str(row.get("anchor", ""))) + len(str(row.get("positive", "")))


def _length_sorted(stream: Iterator[Any], window: int) -> Iterator[Any]:
    """Group a stream so that nearby rows have similar text lengths.

    Collators pad every batch to its longest row, so a random batch pays for a
    length outlier.  Sorting inside a sliding window keeps each batch's rows
    close in length, which cuts padding dramatically on long-tailed text.  The
    window is small relative to the shuffle buffer, so global order stays
    random and the data distribution does not drift across an epoch.
    """

    window = max(2, int(window))
    chunk: list[Any] = []
    for item in stream:
        chunk.append(item)
        if len(chunk) >= window:
            chunk.sort(key=_text_length)
            yield from chunk
            chunk = []
    if chunk:
        chunk.sort(key=_text_length)
        yield from chunk


class ParquetTripletDataset(IterableDataset[dict[str, str]]):
    """Stream triplets from one parquet file or a directory of shards.

    Rows are split deterministically first and then sharded across torchrun
    ranks.  Only the current Arrow record batch is resident in memory.
    """

    def __init__(
        self,
        path: str,
        columns: list[str],
        max_samples: int | None = None,
        split: str = "train",
        split_modulo: int = 20,
        shuffle: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 42,
        length_sort_window: int = 0,
    ) -> None:
        self.path = Path(path)
        self.columns = columns
        self.max_samples = max_samples
        self.split = split
        self.split_modulo = split_modulo
        self.shuffle = shuffle
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.seed = int(seed)
        self.length_sort_window = int(length_sort_window or 0)
        self._epoch = 0

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        return sorted(self.path.glob("*.parquet"))

    def _iter_rows(self) -> Iterator[dict[str, str]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - environment error
            raise RuntimeError("pyarrow is required for parquet retrieval data") from exc

        rank, world_size = _shard_context()
        seen = 0
        accepted = 0
        yielded = 0
        for file_path in self._files():
            parquet = pq.ParquetFile(file_path)
            for batch in parquet.iter_batches(columns=self.columns, batch_size=256):
                for row in batch.to_pylist():
                    row_index = seen
                    seen += 1
                    if not split_accepts(row_index, self.split, self.split_modulo):
                        continue
                    if accepted % world_size != rank:
                        accepted += 1
                        continue
                    accepted += 1
                    yield {key: str(value or "") for key, value in row.items()}
                    yielded += 1
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return

    def set_epoch(self, epoch: int) -> None:
        """Seed the shuffle stream for a new epoch.

        The training loop calls this on the parent dataset before iterating;
        DataLoader workers inherit the value when they are forked, so the
        shuffle order can change every epoch even with multiple workers.
        """

        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, str]]:
        rank, _ = _shard_context()
        stream = _buffered_shuffle(
            self._iter_rows(),
            self.shuffle_buffer if self.shuffle else 0,
            self.seed + self._epoch * 7919 + rank,
        )
        if self.length_sort_window > 0:
            return _length_sorted(stream, self.length_sort_window)
        return stream


class ClipTarDataset(IterableDataset[dict[str, Any]]):
    """Stream ``*.jpg``/``*.txt`` pairs from COCO WebDataset tar shards."""

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
        split: str = "train",
        split_modulo: int = 20,
        shuffle: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 42,
    ) -> None:
        self.path = Path(path)
        self.max_samples = max_samples
        self.split = split
        self.split_modulo = split_modulo
        self.shuffle = shuffle
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.seed = int(seed)
        self._epoch = 0

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        return sorted(self.path.glob("*.tar"))

    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        rank, world_size = _shard_context()
        seen = 0
        accepted = 0
        yielded = 0
        for shard in self._files():
            with tarfile.open(shard, mode="r:") as archive:
                members = {member.name: member for member in archive if member.isfile()}
                for name, member in sorted(members.items()):
                    if not name.endswith(".jpg"):
                        continue
                    text_member = members.get(name[:-4] + ".txt")
                    if text_member is None:
                        continue
                    row_index = seen
                    seen += 1
                    if not split_accepts(row_index, self.split, self.split_modulo):
                        continue
                    if accepted % world_size != rank:
                        accepted += 1
                        continue
                    accepted += 1
                    image_file = archive.extractfile(member)
                    text_file = archive.extractfile(text_member)
                    if image_file is None or text_file is None:
                        continue
                    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
                    text = text_file.read().decode("utf-8", errors="replace").strip()
                    if not text:
                        continue
                    yield {"image": image, "text": text}
                    yielded += 1
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return

    def set_epoch(self, epoch: int) -> None:
        """Seed the shuffle stream for a new epoch (see ParquetTripletDataset)."""

        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rank, _ = _shard_context()
        # Images are large, so the clip/vl configs use a smaller buffer than the
        # text datasets.
        return _buffered_shuffle(
            self._iter_rows(),
            self.shuffle_buffer if self.shuffle else 0,
            self.seed + self._epoch * 7919 + rank,
        )


def retrieval_collate(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Keep strings and PIL images as lists instead of tensor-collating them."""

    if not rows:
        return {}
    return {key: [row[key] for row in rows] for key in rows[0]}


def make_loader(
    dataset: IterableDataset,
    batch_size: int,
    pin_memory: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create the standard loader for a streaming retrieval dataset.

    Workers only see distinct samples because the datasets shard by
    ``(rank, worker_id)``; the persistent-worker flag is left off so each
    epoch re-enters ``__iter__`` and reshuffles.
    """

    num_workers = max(0, int(num_workers))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=retrieval_collate,
        persistent_workers=False,
    )


def build_dataset(
    task: str,
    data_cfg: DictConfig,
    path: str,
    split: str,
    max_samples: int | None,
) -> IterableDataset:
    """Build one dataset according to the task and configured path."""

    split_modulo = int(data_cfg.get("split_modulo", 20))
    shuffle_buffer = int(data_cfg.get("shuffle_buffer", 0) or 0)
    seed = int(data_cfg.get("shuffle_seed", data_cfg.get("seed", 42)) or 42)
    # Only the training split is shuffled: evaluation batches must stay
    # deterministic and comparable across epochs.
    shuffle = str(split).lower() in {"train", "training"} and shuffle_buffer > 0
    length_sort_window = int(data_cfg.get("length_sort_window", 0) or 0)
    if not shuffle:
        length_sort_window = 0
    if task in {"clip", "vl_embedding"}:
        return ClipTarDataset(
            path,
            max_samples,
            split=split,
            split_modulo=split_modulo,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
        )
    columns = ["anchor", "positive"] + [
        f"negative_{index}" for index in range(1, int(data_cfg.get("num_negatives", 1)) + 1)
    ]
    return ParquetTripletDataset(
        path,
        columns,
        max_samples,
        split=split,
        split_modulo=split_modulo,
        shuffle=shuffle,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
        length_sort_window=length_sort_window,
    )


def build_split_loaders(
    task: str,
    data_cfg: DictConfig,
    batch_size: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation and test loaders from the native config."""

    train_path = str(data_cfg.get("train_path", data_cfg.get("path")))
    validation_path = str(data_cfg.get("validation_path", train_path))
    test_path = str(data_cfg.get("test_path", train_path))
    train_split = str(data_cfg.get("train_split", "train"))
    validation_split = str(data_cfg.get("validation_split", "validation"))
    test_split = str(data_cfg.get("test_split", "test"))
    train_max = data_cfg.get("max_samples")
    validation_max = data_cfg.get("validation_max_samples", data_cfg.get("eval_max_samples"))
    test_max = data_cfg.get("test_max_samples", data_cfg.get("eval_max_samples"))
    datasets = (
        build_dataset(task, data_cfg, train_path, train_split, train_max),
        build_dataset(task, data_cfg, validation_path, validation_split, validation_max),
        build_dataset(task, data_cfg, test_path, test_split, test_max),
    )
    # Training may parallelize decoding across workers; evaluation stays
    # single-process so that ``*_max_samples`` counts real samples and results
    # stay bit-for-bit comparable across runs.
    train_workers = int(data_cfg.get("num_workers", 0) or 0)
    loaders = (
        make_loader(datasets[0], batch_size, pin_memory, train_workers),
        make_loader(datasets[1], batch_size, pin_memory, 0),
        make_loader(datasets[2], batch_size, pin_memory, 0),
    )
    return loaders


__all__ = [
    "ClipTarDataset",
    "ParquetTripletDataset",
    "build_dataset",
    "build_split_loaders",
    "make_loader",
    "retrieval_collate",
    "split_accepts",
]
