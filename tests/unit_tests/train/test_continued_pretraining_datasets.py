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

"""Readers for continued-pretraining corpora (sentence pairs, parquet images).

These datasets differ from the task-training ones in a way that is easy to get
wrong and hard to notice: the sentence-pair sets are *balanced* positive/negative,
so reading every row inverts most of the labels, and the image sets interleave
several captions per image, so splitting by row index leaks captions across
splits.  Both cases still produce a finite loss, which is why they need tests.
"""

import io

import pytest

pytest.importorskip("torch")
pyarrow = pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")
pytest.importorskip("PIL")

import pyarrow.parquet as pq  # noqa: E402
from PIL import Image  # noqa: E402

from flagscale.train.datasets.retrieval import (  # noqa: E402
    ClipParquetDataset,
    SentencePairDataset,
)


def _write_pairs(path, rows) -> None:
    pq.write_table(
        pyarrow.table(
            {
                "sentence1": [r[0] for r in rows],
                "sentence2": [r[1] for r in rows],
                "score": [str(r[2]) for r in rows],
            }
        ),
        path,
    )


def test_sentence_pairs_find_shards_under_data_dir(tmp_path) -> None:
    """HuggingFace-style exports keep shards in ``data/``; flat layouts also exist.

    ModelScope datasets (LCQMC, Flickr30K) unpack into ``<repo>/data/*.parquet``
    while the T2Ranking dump is flat.  A reader that only globs the top level
    silently yields zero rows on the downloaded layout -- an empty epoch, which
    looks like "training finished instantly" rather than an error.
    """

    for layout in ("flat", "nested"):
        root = tmp_path / layout
        target = root / "data" if layout == "nested" else root
        target.mkdir(parents=True)
        _write_pairs(target / "train-0.parquet", [("甲", "甲的改写", 1)])

        got = list(SentencePairDataset(str(root), split="all"))
        assert len(got) == 1, f"{layout} layout found {len(got)} rows"


def test_clip_parquet_find_shards_under_data_dir(tmp_path) -> None:
    for layout in ("flat", "nested"):
        root = tmp_path / f"clip-{layout}"
        target = root / "data" if layout == "nested" else root
        target.mkdir(parents=True)
        pq.write_table(
            pyarrow.table(
                {
                    "image": [{"bytes": _png_bytes("red"), "path": None}],
                    "caption": [f"caption-{layout}"],
                }
            ),
            target / "test-0.parquet",
        )

        got = list(ClipParquetDataset(str(root), split="all"))
        assert len(got) == 1, f"{layout} layout found {len(got)} rows"


def test_clip_parquet_takes_one_caption_from_a_list(tmp_path) -> None:
    """Flickr30K stores 5 captions per image in a ``list<string>`` column.

    ``str()`` on that cell yields ``"['a', 'b', 'c']"``, so the tokenizer would
    train on bracket-and-quote syntax instead of the sentence -- a bug with no
    error, only a quietly worse model.  Exactly one caption must survive.
    """

    path = tmp_path / "multicap.parquet"
    pq.write_table(
        pyarrow.table(
            {
                "image": [{"bytes": _png_bytes("red"), "path": None}],
                "caption": [["第一句描述", "第二句描述", "第三句描述"]],
            }
        ),
        path,
    )

    got = list(ClipParquetDataset(str(path), split="all"))
    assert len(got) == 1
    assert got[0]["text"] == "第一句描述"
    assert "[" not in got[0]["text"] and "'" not in got[0]["text"]


def test_clip_parquet_skips_empty_caption_lists(tmp_path) -> None:
    path = tmp_path / "emptycap.parquet"
    pq.write_table(
        pyarrow.table(
            {
                "image": [{"bytes": _png_bytes("red"), "path": None}],
                "caption": [[]],
            }
        ),
        path,
    )
    assert list(ClipParquetDataset(str(path), split="all")) == []


def test_split_named_shards_are_not_mixed(tmp_path) -> None:
    """A dataset's own split files must be respected, not pooled and re-bucketed.

    LCQMC ships ``train-*.parquet`` / ``validation-*.parquet`` / ``test-*.parquet``.
    Globbing all of them and re-splitting with the hash rule put **3,046 sentences
    that exist only in LCQMC's official test file** into our training split.  The
    published comparisons were all cross-dataset so the numbers stood, but the
    moment evaluation reuses the training corpus this is a silent leak.
    """

    for split in ("train", "validation", "test"):
        target = tmp_path / "data"
        target.mkdir(exist_ok=True)
        # Enough rows that the hash bucketing (modulo 20) has something to keep
        # in every bucket; one row per file would land wherever its hash falls.
        _write_pairs(
            target / f"{split}-00000-of-00001-abc.parquet",
            [(f"{split}-only-{i}", f"{split}-only-pos-{i}", 1) for i in range(200)],
        )

    # `split="all"` means "no split filtering" in the hash rule; the file filter
    # must not narrow it, otherwise an evaluator asking for everything sees one
    # third of the data.
    everybody = {row["anchor"] for row in SentencePairDataset(str(tmp_path), split="all")}
    assert all(any(a.startswith(p) for a in everybody) for p in ("train-", "validation-", "test-"))

    for split in ("train", "validation", "test"):
        rows = {row["anchor"] for row in SentencePairDataset(str(tmp_path), split=split)}
        offenders = {a for a in rows if not a.startswith(f"{split}-")}
        assert not offenders, f"{split} split leaked from another file: {sorted(offenders)[:3]}"
        assert rows, f"{split} split is empty; the filter dropped everything"


def test_unnamed_shards_still_use_the_hash_split(tmp_path) -> None:
    """A single unnamed dump (T2Ranking, MS MARCO triplets) must keep working.

    Those files carry no split prefix, so the filter has nothing to match and the
    old hash-based partition is the only partitioning there is.
    """

    path = tmp_path / "train-00000-of-00011.parquet"
    _write_pairs(path, [(f"q{i}", f"p{i}", 1) for i in range(60)])

    train = sum(1 for _ in SentencePairDataset(str(path), split="train"))
    validation = sum(1 for _ in SentencePairDataset(str(path), split="validation"))
    test = sum(1 for _ in SentencePairDataset(str(path), split="test"))
    # T2Ranking's shards are named `train-*` too, so this one *is* filtered to
    # train.  What matters is that the hash split still partitions inside it.
    assert train > 0
    assert train + validation + test == 60, (train, validation, test)


def test_sentence_pairs_keep_only_positive_samples(tmp_path) -> None:
    """Negative pairs must be dropped, not relabelled as positives.

    The published sets are ~50/50, so a reader that ignores `score` would train
    the encoder to pull apart the pairs it should pull together.
    """

    rows = [
        ("如何学习编程", "怎样学习编程", 1),
        ("如何学习编程", "今天天气不错", 0),
        ("北京是首都", "北京是中国的首都", 1),
        ("苹果手机", "苹果好吃", 0),
        ("深度学习入门", "机器学习入门", 1),
    ]
    path = tmp_path / "pairs.parquet"
    _write_pairs(path, rows)

    dataset = SentencePairDataset(
        str(path), split="all", columns=("sentence1", "sentence2")
    )
    got = list(dataset)

    assert len(got) == 3, f"expected the 3 positive pairs, got {len(got)}"
    assert {row["anchor"] for row in got} == {"如何学习编程", "北京是首都", "深度学习入门"}
    # The schema is renamed so `embedding_loss` (which reads anchor/positive)
    # works unchanged -- that is the whole point of this adapter.
    assert all(set(row) == {"anchor", "positive"} for row in got)


def test_sentence_pairs_skip_blank_sides(tmp_path) -> None:
    """A blank side would encode to a degenerate vector; it must be skipped."""

    path = tmp_path / "blank.parquet"
    _write_pairs(path, [("有内容", "", 1), ("正常", "正常对", 1)])
    got = list(SentencePairDataset(str(path), split="all"))
    assert len(got) == 1
    assert got[0]["anchor"] == "正常"


def test_sentence_pairs_respect_max_samples(tmp_path) -> None:
    path = tmp_path / "many.parquet"
    _write_pairs(path, [(f"a{i}", f"b{i}", 1) for i in range(50)])
    got = list(SentencePairDataset(str(path), max_samples=7, split="all"))
    assert len(got) == 7


def _png_bytes(color) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_clip_parquet_decodes_image_struct(tmp_path) -> None:
    """The ``datasets`` Image column is a struct with a ``bytes`` field."""

    path = tmp_path / "clip.parquet"
    pq.write_table(
        pyarrow.table(
            {
                "image": [{"bytes": _png_bytes("red"), "path": None}],
                "caption": ["一只红色的方块"],
            }
        ),
        path,
    )

    got = list(ClipParquetDataset(str(path), split="all"))
    assert len(got) == 1
    assert isinstance(got[0]["image"], Image.Image)
    assert got[0]["image"].mode == "RGB"
    assert got[0]["image"].size == (4, 4)
    assert got[0]["text"] == "一只红色的方块"


def test_clip_parquet_accepts_raw_bytes(tmp_path) -> None:
    """A plain binary column is also valid; some exporters write it that way."""

    path = tmp_path / "raw.parquet"
    pq.write_table(
        pyarrow.table({"image": [_png_bytes("blue")], "caption": ["蓝色"]}),
        path,
    )

    got = list(ClipParquetDataset(str(path), split="all"))
    assert len(got) == 1
    assert got[0]["image"].size == (4, 4)


def test_clip_parquet_skips_undecodable_rows(tmp_path) -> None:
    """One corrupt image must not abort a multi-hour run."""

    path = tmp_path / "corrupt.parquet"
    pq.write_table(
        pyarrow.table(
            {
                "image": [
                    {"bytes": b"not-an-image", "path": None},
                    {"bytes": _png_bytes("green"), "path": None},
                ],
                "caption": ["坏的", "好的"],
            }
        ),
        path,
    )

    got = list(ClipParquetDataset(str(path), split="all"))
    assert len(got) == 1
    assert got[0]["text"] == "好的"


def test_clip_parquet_splits_by_caption_not_row_index(tmp_path) -> None:
    """The same caption must never appear on both sides of a split.

    Flickr30K gives each image several captions; splitting by row index puts
    some captions of one group into train and others into eval.
    """

    captions = ["同一句描述"] * 20
    path = tmp_path / "repeat.parquet"
    pq.write_table(
        pyarrow.table(
            {
                "image": [{"bytes": _png_bytes("gray"), "path": None}] * 20,
                "caption": captions,
            }
        ),
        path,
    )

    train = {row["text"] for row in ClipParquetDataset(str(path), split="train", split_modulo=20)}
    validation = {
        row["text"] for row in ClipParquetDataset(str(path), split="validation", split_modulo=20)
    }
    # Grouping on the caption sends the whole group one way, so at most one side
    # is non-empty. The failure this guards against is both sides getting rows.
    assert not (train and validation), (
        f"caption leaked across splits: {len(train)} in train, {len(validation)} in validation"
    )
