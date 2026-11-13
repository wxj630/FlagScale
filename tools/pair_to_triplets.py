#!/usr/bin/env python3
"""Turn a sentence-pair file into reranker/embedding triplets for same-corpus eval.

**Why this exists.** Evaluating continued pretraining on the corpus it was trained
on is the honest question ("did this corpus help on this corpus?"), but the
sentence-pair datasets (LCQMC and friends) ship only `sentence1/sentence2/score`.
`evaluate_retrieval.py` needs `anchor` + `positive` + `negative_*`: without
negatives each query's candidate pool is a single item and R@1 is 1.0 by
construction, which measures nothing.

Negatives are sampled uniformly from the same corpus, excluding anything that
appears as a positive for that anchor.  That is the weak-but-standard setup for a
corpus with no mined hard negatives; it is recorded here so a reader knows the
difficulty comes from random distractors, not from hard mining.

**Split safety.** The reader applies our hash bucketing on top of whatever files
it is given, so this script only materialises ONE split's file.  Point
`--input` at the dataset's own `test-*.parquet` to get a test set that is
disjoint from training by construction.

Usage::

    python tools/pair_to_triplets.py \
        --input /flagos-retrieval-datasets/lcqmc/data/test-00000-of-00001-*.parquet \
        --out /flagos-retrieval-datasets/lcqmc/triplets-test \
        --num-negatives 15 --max-positives 4000 --seed 42
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="parquet file or glob")
    parser.add_argument("--out", required=True)
    parser.add_argument("--a-column", default="sentence1")
    parser.add_argument("--b-column", default="sentence2")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--num-negatives", type=int, default=15)
    parser.add_argument("--max-positives", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise SystemExit(f"no files match {args.input}")

    rows: list[dict] = []
    for path in paths:
        table = pq.read_table(
            path, columns=[args.a_column, args.b_column, args.score_column]
        )
        rows.extend(table.to_pylist())
    print(f"read {len(rows)} rows from {len(paths)} file(s)")

    def positive(row: dict) -> bool:
        try:
            return float(row[args.score_column]) == 1.0
        except (TypeError, ValueError):
            return False

    pairs = [
        (str(r[args.a_column]).strip(), str(r[args.b_column]).strip())
        for r in rows
        if positive(r)
        and str(r[args.a_column]).strip()
        and str(r[args.b_column]).strip()
    ]
    print(f"positive pairs: {len(pairs)}")
    if not pairs:
        raise SystemExit("no positive pairs; check --score-column")

    # The distractor pool is every *positive* sentence in the corpus, not every
    # sentence.  Sampling uniformly from all sentences makes the task trivial:
    # measured on LCQMC, base R@1 came out at 0.9992 because a random unrelated
    # question is never confused with the anchor, so the metric had no headroom
    # and could not show any effect.  Positives of other anchors are all
    # plausible answers to *some* question, which is the same construction the
    # T2Ranking triplets use.
    pool = sorted({pair[1] for pair in pairs})
    print(f"distractor pool: {len(pool)} distinct sentences")

    rng = random.Random(args.seed)
    # A deterministic shuffle decides which positives survive --max-positives,
    # so two conversions of the same input produce the same evaluation set.
    order = list(range(len(pairs)))
    rng.shuffle(order)
    order = order[: args.max_positives]

    # Anchors that appear with several positives must treat all of them as
    # non-negatives, or the eval would score a valid answer as a distractor.
    positives_by_anchor: dict[str, set[str]] = {}
    for anchor, pos in pairs:
        positives_by_anchor.setdefault(anchor, set()).add(pos)

    out_rows: list[dict] = []
    skipped = 0
    for index in order:
        anchor, pos = pairs[index]
        banned = positives_by_anchor[anchor] | {anchor}
        negatives: list[str] = []
        for _ in range(args.num_negatives * 8):
            if len(negatives) >= args.num_negatives:
                break
            candidate = pool[rng.randrange(len(pool))]
            if candidate in banned or candidate in negatives:
                continue
            negatives.append(candidate)
        if len(negatives) < args.num_negatives:
            skipped += 1
            continue
        row = {"anchor": anchor, "positive": pos}
        for i, neg in enumerate(negatives, start=1):
            row[f"negative_{i}"] = neg
        out_rows.append(row)

    columns = ["anchor", "positive"] + [
        f"negative_{i}" for i in range(1, args.num_negatives + 1)
    ]
    table = pa.table({name: [r[name] for r in out_rows] for name in columns})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / "triplets.parquet"
    pq.write_table(table, destination)
    print(
        f"wrote {len(out_rows)} triplets ({skipped} skipped) -> {destination}\n"
        "  point data.test_path at this DIRECTORY and set test_format: triplet"
    )


if __name__ == "__main__":
    main()
