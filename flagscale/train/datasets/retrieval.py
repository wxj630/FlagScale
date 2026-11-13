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

import hashlib
import io
import os
import random
import tarfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


def _filter_by_split(files: list[Path], split: str) -> list[Path]:
    """Drop a dataset's own other-split shards when its files declare one.

    HuggingFace/ModelScope exports name shards after the split they belong to
    (``train-00000-of-00001-<hash>.parquet``, ``test-...``).  Globbing every
    parquet and then re-bucketing with ``split_accepts`` silently merges those
    splits: measured on LCQMC, our 'train' split contained **3,046 sentences that
    exist only in LCQMC's official test file**, and validation was not a clean
    holdout either.

    That did not corrupt the comparisons in docs/retrieval/training/22 (all of
    them cross datasets: LCQMC -> T2Ranking, MS MARCO -> T2Ranking, Flickr30K ->
    MSCOCO), but it *would* leak the moment anyone evaluates on the same corpus
    they trained on -- which is the normal case for this pipeline.

    Only files whose name starts with a known split token are filtered.  A dump
    with a single unnamed shard (T2Ranking, our MS MARCO triplets) has no split
    prefix, so nothing is dropped and the hash split keeps working as before.
    """

    known = {"train", "training", "validation", "valid", "val", "test", "testing"}
    prefixed = []
    for path in files:
        stem = path.stem.split("-")[0].lower()
        if stem in known:
            prefixed.append((stem, path))
    if not prefixed:
        return files

    wanted = {str(split).lower()}
    if wanted & {"train", "training"}:
        wanted |= {"train", "training"}
    elif wanted & {"validation", "valid", "val"}:
        wanted |= {"validation", "valid", "val"}
    elif wanted & {"test", "testing"}:
        wanted |= {"test", "testing"}
    else:
        return files

    selected = [path for stem, path in prefixed if stem in wanted]
    # An unknown/absent split name must not silently yield zero rows; fall back
    # to every file so the hash split still produces data and the caller's own
    # emptiness check reports a real problem if it happens.
    return selected or files


def split_accepts(
    index: int,
    split: str,
    modulo: int = 20,
    group_key: str | None = None,
) -> bool:
    """Return whether a deterministic row belongs to ``split``.

    T2Ranking ships only train shards.  For the native retrieval example we
    reserve 90%/5%/5% of rows for train/validation/test without materializing
    another copy of the dataset.  COCO's separately supplied test directory
    uses the ``all`` split and therefore bypasses this partition.

    ``group_key`` makes the partition ignore ``index`` and bucket on a stable
    hash of it instead.  T2Ranking answers one query with several relevant
    passages, so a single query appears on ~14 rows; bucketing by row index
    scattered those rows across splits and 89.4% of the test queries had also
    been seen in training.  Passing the query text as ``group_key`` keeps every
    row of a query in the same split, which is the only way the reported number
    measures retrieval rather than memorisation.
    """

    split = str(split).lower()
    if split in {"all", "none"}:
        return True
    modulo = max(3, int(modulo))
    if group_key is None:
        bucket = index % modulo
    else:
        # hashlib rather than hash(): the built-in is salted per process, so the
        # split would differ between runs and between the training and
        # evaluation processes, which is exactly the reproducibility this
        # module is supposed to guarantee.
        digest = hashlib.blake2b(group_key.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, "big") % modulo
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
    """Combine the data-parallel rank and DataLoader worker id into one shard.

    An ``IterableDataset`` is materialized independently in every worker, so
    without worker-aware sharding each of the ``num_workers`` processes would
    yield the full stream and duplicate samples.  Pairing rank with worker id
    gives every (rank, worker) a distinct slice of the split.

    The rank used here must be the *data-parallel* rank, not the global
    torchrun rank.  With pipeline parallelism every stage of the same pipeline
    has to consume the same samples -- stage1 scores the activations stage0
    produced.  Sharding by global rank gave the two stages disjoint data, so
    stage1 paired a hidden state with another sample's labels: measured on
    P800, the same prompt scored 4.55 through p=2 versus 1.25 through HF, while
    the identical weights in a single stage scored 1.31 (correct).  Falls back
    to the global rank when Megatron's parallel state is not initialised.
    """

    rank, world_size = _distributed_context()
    try:
        from megatron.core import parallel_state as _ps

        if _ps.model_parallel_is_initialized():
            # dp rank repeats across pipeline stages; dp size is what shards.
            rank = _ps.get_data_parallel_rank()
            world_size = _ps.get_data_parallel_world_size()
    except Exception:
        pass
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
                    # Bucket on the query, not the row: one t2ranking query
                    # answers through ~14 rows, and splitting those by row
                    # index leaked 89% of the test queries into training.
                    if not split_accepts(
                        row_index,
                        self.split,
                        self.split_modulo,
                        group_key=str(row.get("anchor", "")),
                    ):
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


class SentencePairDataset(IterableDataset[dict[str, str]]):
    """Stream ``(sentence1, sentence2, score)`` pairs as anchor/positive.

    Continued pretraining needs *contrastive* pairs, not labelled triplets: the
    encoder learns by pulling a paraphrase together and pushing unrelated
    in-batch sentences apart.  The MTEB/CMTEB sentence-pair sets are exactly that
    shape, so this reads them directly and renames the columns instead of asking
    the caller to rewrite the files.

    **Only ``score == 1`` rows are emitted.**  These sets are balanced
    positive/negative pairs (LCQMC 58% positive, BQ exactly 50%, PAWSX 44%,
    AFQMC 31%), so feeding every row would train the encoder to pull together
    sentences it should separate -- the label would be silently inverted for
    most of the corpus.  Dropping the negative rows also keeps the meaning of
    ``positive`` honest, which the contrastive loss relies on.
    """

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
        split: str = "train",
        split_modulo: int = 20,
        shuffle: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 42,
        length_sort_window: int = 0,
        columns: tuple[str, str] = ("sentence1", "sentence2"),
        score_column: str = "score",
        positive_value: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.max_samples = max_samples
        self.split = split
        self.split_modulo = split_modulo
        self.shuffle = shuffle
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.seed = int(seed)
        self.length_sort_window = int(length_sort_window or 0)
        self.columns = columns
        self.score_column = score_column
        self.positive_value = positive_value
        self._epoch = 0

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        # HuggingFace-style exports put shards in `data/`; the T2Ranking dump puts
        # them at the top level.  Check both so neither layout needs a re-pack.
        nested = sorted(self.path.glob("data/*.parquet"))
        candidates = nested or sorted(self.path.glob("*.parquet"))
        return _filter_by_split(candidates, self.split)

    def _is_positive(self, raw: object) -> bool:
        # The score arrives as a string from dataset_infos-declared "Value"
        # columns, so compare numerically rather than by identity.
        try:
            return float(raw) == float(self.positive_value)
        except (TypeError, ValueError):
            return False

    def _iter_rows(self) -> Iterator[dict[str, str]]:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyarrow is required for parquet retrieval data") from exc

        rank, world_size = _shard_context()
        first, second = self.columns
        wanted = [first, second, self.score_column]
        seen = 0
        accepted = 0
        yielded = 0
        for file_path in self._files():
            parquet = pq.ParquetFile(file_path)
            for batch in parquet.iter_batches(columns=wanted, batch_size=256):
                for row in batch.to_pylist():
                    row_index = seen
                    seen += 1
                    if not self._is_positive(row.get(self.score_column)):
                        continue
                    anchor = str(row.get(first) or "").strip()
                    positive = str(row.get(second) or "").strip()
                    # A blank side would make the pair encode to a degenerate
                    # vector and contribute a meaningless contrastive term.
                    if not anchor or not positive:
                        continue
                    # Bucket on the anchor so one sentence cannot land in both
                    # train and evaluation splits (same rule as the triplets).
                    if not split_accepts(
                        row_index, self.split, self.split_modulo, group_key=anchor
                    ):
                        continue
                    if accepted % world_size != rank:
                        accepted += 1
                        continue
                    accepted += 1
                    yield {"anchor": anchor, "positive": positive}
                    yielded += 1
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return

    def set_epoch(self, epoch: int) -> None:
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


class ClipParquetDataset(IterableDataset[dict[str, Any]]):
    """Stream ``(image, caption)`` pairs from a HuggingFace-style parquet export.

    ``ClipTarDataset`` reads WebDataset tars because the COCO shards we started
    with were packaged that way, but the tar layout is an accident of that dump,
    not a requirement: the trainer only needs an image and its caption.  Datasets
    published only as parquet (Flickr30K on ModelScope puts a raw image column
    next to a caption column) are equally usable, so read them directly instead
    of asking for a re-pack.

    Two column types are accepted for the image: a struct with ``bytes``/``path``
    (the ``datasets`` Image feature) or a plain ``bytes`` blob.
    """

    def __init__(
        self,
        path: str,
        max_samples: int | None = None,
        split: str = "train",
        split_modulo: int = 20,
        shuffle: bool = False,
        shuffle_buffer: int = 0,
        seed: int = 42,
        image_column: str = "image",
        text_column: str = "caption",
    ) -> None:
        self.path = Path(path)
        self.max_samples = max_samples
        self.split = split
        self.split_modulo = split_modulo
        self.shuffle = shuffle
        self.shuffle_buffer = int(shuffle_buffer or 0)
        self.seed = int(seed)
        self.image_column = image_column
        self.text_column = text_column
        self._epoch = 0

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        # Flickr30K from ModelScope ships shards under `data/`; check the flat
        # layout too so a re-packed export also works.
        nested = sorted(self.path.glob("data/*.parquet"))
        candidates = nested or sorted(self.path.glob("*.parquet"))
        return _filter_by_split(candidates, self.split)

    def _pick_text(self, value: Any) -> str:
        """Normalise a caption cell to one plain string.

        Flickr30K's ``caption`` column is ``list<string>`` (5 captions per image).
        Calling ``str()`` on that list -- which the first version did -- yields the
        Python repr ``"['a', 'b', 'c']"``, so the tokenizer was trained on
        bracket-and-quote syntax instead of the sentence.  Nothing raises: the loss
        still falls, the model just learns the wrong text.  Take the first caption
        and keep the rest out of the training signal.
        """

        if isinstance(value, (list, tuple)):
            for item in value:
                text = str(item or "").strip()
                if text:
                    return text
            return ""
        return str(value or "").strip()

    def _decode_image(self, value: Any) -> Any:
        if isinstance(value, dict):
            payload = value.get("bytes")
            if payload is None and value.get("path"):
                # `datasets` stores a path when the image was not embedded.
                with open(value["path"], "rb") as handle:
                    payload = handle.read()
        else:
            payload = value
        if payload is None:
            raise ValueError("image column has neither bytes nor a readable path")
        return Image.open(io.BytesIO(payload)).convert("RGB")

    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyarrow is required for parquet retrieval data") from exc

        rank, world_size = _shard_context()
        seen = 0
        accepted = 0
        yielded = 0
        for file_path in self._files():
            parquet = pq.ParquetFile(file_path)
            for batch in parquet.iter_batches(
                columns=[self.image_column, self.text_column], batch_size=64
            ):
                for row in batch.to_pylist():
                    row_index = seen
                    seen += 1
                    text = self._pick_text(row.get(self.text_column))
                    if not text:
                        continue
                    # Flickr30K ships several captions per image in ONE row (the
                    # column is list<string>), so there is no row-level leakage
                    # to guard against here -- one row is one (image, caption)
                    # sample.  Bucket on the caption text so two rows carrying the
                    # same sentence cannot straddle the train/eval boundary.
                    if not split_accepts(
                        row_index, self.split, self.split_modulo, group_key=text
                    ):
                        continue
                    if accepted % world_size != rank:
                        accepted += 1
                        continue
                    accepted += 1
                    try:
                        image = self._decode_image(row.get(self.image_column))
                    except Exception:
                        # One corrupt JPEG must not abort a multi-hour run.
                        continue
                    yield {"image": image, "text": text}
                    yielded += 1
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        rank, _ = _shard_context()
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


def _worker_init(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically.

    Offsets are relative to whatever ``seed_everything`` set in the parent, so a
    rerun with the same ``train.seed`` reproduces the same per-worker streams.
    Without this, worker RNG state depends on OS scheduling and a "same seed"
    rerun diverges as soon as ``num_workers > 0``.
    """

    import random as _random

    import numpy as _np
    import torch as _torch

    base = _torch.initial_seed() % (2**32)
    worker_seed = (base + worker_id * 7919) % (2**32)
    _random.seed(worker_seed)
    _np.random.seed(worker_seed)
    _torch.manual_seed(worker_seed)


def make_loader(
    dataset: IterableDataset,
    batch_size: int,
    pin_memory: bool,
    num_workers: int = 0,
    seed: int | None = None,
) -> DataLoader:
    """Create the standard loader for a streaming retrieval dataset.

    Workers only see distinct samples because the datasets shard by
    ``(rank, worker_id)``; the persistent-worker flag is left off so each
    epoch re-enters ``__iter__`` and reshuffles.

    ``seed`` fixes the sampler/worker RNG.  Passing it makes ``num_workers > 0``
    runs repeatable; leaving it ``None`` keeps the previous behaviour.
    """

    num_workers = max(0, int(num_workers))
    generator = None
    worker_init = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed) % (2**63 - 1))
        worker_init = _worker_init
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=retrieval_collate,
        persistent_workers=False,
        generator=generator,
        worker_init_fn=worker_init,
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
        # `data.format` lets the same task read either packaging: the COCO dump
        # is WebDataset tar, Flickr30K on ModelScope is HuggingFace parquet.
        image_format = str(data_cfg.get("format", "tar")).lower()
        if image_format == "parquet":
            return ClipParquetDataset(
                path,
                max_samples,
                split=split,
                split_modulo=split_modulo,
                shuffle=shuffle,
                shuffle_buffer=shuffle_buffer,
                seed=seed,
                image_column=str(data_cfg.get("image_column", "image")),
                text_column=str(data_cfg.get("text_column", "caption")),
            )
        return ClipTarDataset(
            path,
            max_samples,
            split=split,
            split_modulo=split_modulo,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
        )
    # Sentence-pair corpora (continued-pretraining data) carry no negatives, so
    # the triplet reader would KeyError on `negative_1`.  Dispatch on the
    # declared format rather than probing, so a misconfiguration fails at build
    # time with the name of the setting instead of deep inside a worker.
    text_format = str(data_cfg.get("format", "triplet")).lower()
    if text_format in {"pair", "sentence_pair"}:
        columns = data_cfg.get("pair_columns", ["sentence1", "sentence2"])
        return SentencePairDataset(
            path,
            max_samples,
            split=split,
            split_modulo=split_modulo,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
            length_sort_window=length_sort_window,
            columns=(str(columns[0]), str(columns[1])),
            score_column=str(data_cfg.get("score_column", "score")),
            positive_value=float(data_cfg.get("positive_value", 1.0)),
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
    seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation and test loaders from the native config.

    ``seed`` is the training seed; it fixes the sampler and worker RNG so a run
    is repeatable with ``num_workers > 0``.  Evaluation loaders are left
    unseeded on purpose -- their order is already fixed by the dataset split, and
    a generator there would change the sample order of an otherwise identical
    rerun.
    """

    train_path = str(data_cfg.get("train_path", data_cfg.get("path")))
    validation_path = str(data_cfg.get("validation_path", train_path))
    test_path = str(data_cfg.get("test_path", train_path))
    train_split = str(data_cfg.get("train_split", "train"))
    validation_split = str(data_cfg.get("validation_split", "validation"))
    test_split = str(data_cfg.get("test_split", "test"))
    train_max = data_cfg.get("max_samples")
    validation_max = data_cfg.get("validation_max_samples", data_cfg.get("eval_max_samples"))
    test_max = data_cfg.get("test_max_samples", data_cfg.get("eval_max_samples"))

    # Continued-pretraining configs train on one packaging and evaluate on
    # another: the corpus is a sentence-pair or image-caption parquet, while the
    # acceptance test set is the T2Ranking triplet parquet (or COCO tar).
    # A single `format` cannot describe both, and the mismatch fails silently --
    # the reader finds no `sentence1` column, yields nothing, and evaluation dies
    # with "Evaluation dataset is empty for this split" only after a full epoch.
    # `{train,validation,test}_format` let each split declare its own, defaulting
    # to the shared `format`.
    from copy import deepcopy

    base_format = str(data_cfg.get("format", "triplet"))

    def _split_cfg(override_key: str) -> DictConfig:
        if override_key not in data_cfg:
            return data_cfg
        merged = deepcopy(data_cfg)
        merged.format = str(data_cfg.get(override_key))
        return merged

    datasets = (
        build_dataset(task, _split_cfg("train_format"), train_path, train_split, train_max),
        build_dataset(
            task, _split_cfg("validation_format"), validation_path,
            validation_split, validation_max,
        ),
        build_dataset(task, _split_cfg("test_format"), test_path, test_split, test_max),
    )
    # Training may parallelize decoding across workers; evaluation stays
    # single-process so that ``*_max_samples`` counts real samples and results
    # stay bit-for-bit comparable across runs.
    train_workers = int(data_cfg.get("num_workers", 0) or 0)
    loaders = (
        make_loader(datasets[0], batch_size, pin_memory, train_workers, seed=seed),
        make_loader(datasets[1], batch_size, pin_memory, 0),
        make_loader(datasets[2], batch_size, pin_memory, 0),
    )

    # A mis-matched `format` (pair reader pointed at a triplet file, or vice
    # versa) produces an iterable that yields nothing rather than raising.  That
    # used to surface as "Evaluation dataset is empty for this split" at the END
    # of epoch 1, after a full training pass had been spent.  Probe one batch
    # from the evaluation loaders here so the failure costs seconds.
    #
    # Only the evaluation splits are probed: the training loader is an
    # IterableDataset that re-shuffles per item, so pulling a batch from it now
    # would consume part of the stream the trainer is about to use.
    for name, loader in (("validation", loaders[1]), ("test", loaders[2])):
        try:
            next(iter(loader))
        except StopIteration:
            raise RuntimeError(
                f"{name} split produced no batches: check that `data.format` "
                f"(currently {base_format!r}) matches the files under "
                f"{test_path if name == 'test' else validation_path!r}. "
                "Use `validation_format` / `test_format` when the evaluation "
                "set uses a different packaging from the training corpus."
            ) from None
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
