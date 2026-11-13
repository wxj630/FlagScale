#!/usr/bin/env python3
"""Turn an MTEB-style retrieval dataset into the reranker triplet parquet.

The training reader wants one row per (query, positive, negative_1..N).  MTEB and
ModelScope publish the same information split across three tables --
``corpus.jsonl`` (docid -> text), ``queries.jsonl`` (qid -> text) and
``qrels/*.jsonl`` (qid -> docid -> score) -- because that layout is what the
retrieval benchmarks consume.  This bridges the two without touching the reader.

Negatives are sampled from the corpus, excluding the query's own positives.  That
is the standard recipe for turning a retrieval collection into reranker pairs and
it is also what the paper's "hard negative mining" step approximates; the
difficulty here is random, not hard, which is a documented limitation rather than
a property of the data.

Usage::

    python tools/mteb_to_triplets.py \
        --dataset-dir /flagos-retrieval-datasets/msmarco \
        --out /flagos-retrieval-datasets/msmarco/triplets \
        --num-negatives 1 --max-queries 20000 --seed 42

Run it on the machine that holds the downloaded files; it is CPU-only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(dataset_dir: Path) -> dict[str, str]:
    """``docid -> text``.  MTEB puts the title in a separate field; fold it in.

    A reranker sees the passage as it would be presented at inference time, so
    dropping the title would train and evaluate on different text.
    """

    path = dataset_dir / "corpus.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    corpus: dict[str, str] = {}
    for row in read_jsonl(path):
        doc_id = str(row.get("_id", row.get("id", "")))
        if not doc_id:
            continue
        text = str(row.get("text") or "").strip()
        title = str(row.get("title") or "").strip()
        if title and title.lower() not in text.lower():
            text = f"{title} {text}".strip()
        if text:
            corpus[doc_id] = text
    return corpus


def load_queries(dataset_dir: Path) -> dict[str, str]:
    path = dataset_dir / "queries.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    queries: dict[str, str] = {}
    for row in read_jsonl(path):
        qid = str(row.get("_id", row.get("id", "")))
        text = str(row.get("text") or "").strip()
        if qid and text:
            queries[qid] = text
    return queries


def load_qrels(dataset_dir: Path, split: str) -> dict[str, list[str]]:
    """``qid -> [docid, ...]`` keeping only rows with a positive score."""

    path = dataset_dir / "qrels" / f"{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    positives: dict[str, list[str]] = {}
    for row in read_jsonl(path):
        try:
            score = int(row.get("score", 0))
        except (TypeError, ValueError):
            continue
        if score <= 0:
            continue
        qid = str(row.get("query-id", row.get("query_id", "")))
        doc_id = str(row.get("corpus-id", row.get("corpus_id", "")))
        if qid and doc_id:
            positives.setdefault(qid, []).append(doc_id)
    return positives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--qrels-split", default="train")
    parser.add_argument("--num-negatives", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir, args.qrels_split)
    print(f"corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)} queries-with-positives")

    # Deterministic order: a shuffled run would change which queries survive
    # --max-queries and make two conversions incomparable.
    qids = sorted(qid for qid in qrels if qid in queries)
    random.Random(args.seed).shuffle(qids)
    qids = qids[: args.max_queries]

    doc_ids = list(corpus)
    rng = random.Random(args.seed)
    rows: list[dict[str, str]] = []
    skipped = 0
    for qid in qids:
        positives = [d for d in qrels[qid] if d in corpus]
        if not positives:
            skipped += 1
            continue
        # One row per positive, matching the T2Ranking layout the reader assumes.
        for positive_id in positives:
            banned = set(qrels[qid])
            negatives: list[str] = []
            for _ in range(args.num_negatives * 4):
                if len(negatives) >= args.num_negatives:
                    break
                candidate = doc_ids[rng.randrange(len(doc_ids))]
                if candidate in banned or candidate in negatives:
                    continue
                negatives.append(candidate)
            if len(negatives) < args.num_negatives:
                continue
            row = {"anchor": queries[qid], "positive": corpus[positive_id]}
            for index, doc_id in enumerate(negatives, start=1):
                row[f"negative_{index}"] = corpus[doc_id]
            rows.append(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow as pa
    import pyarrow.parquet as pq

    columns = ["anchor", "positive"] + [
        f"negative_{i}" for i in range(1, args.num_negatives + 1)
    ]
    table = pa.table({name: [row[name] for row in rows] for name in columns})
    # Several shards so the reader can stream and shard across ranks; a single
    # file would pin the loader to one rank's worth of IO.
    shards = max(1, min(8, len(rows) // 5000 or 1))
    chunk_size = (len(rows) + shards - 1) // shards
    for shard in range(shards):
        piece = table.slice(shard * chunk_size, chunk_size)
        if piece.num_rows == 0:
            continue
        destination = out_dir / f"train-{shard:05d}-of-{shards:05d}.parquet"
        pq.write_table(piece, destination)
        print(f"  {destination.name}: {piece.num_rows} rows")

    print(f"wrote {len(rows)} triplets ({skipped} queries skipped) to {out_dir}")


if __name__ == "__main__":
    main()
