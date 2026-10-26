"""Verify the retrieval split keeps a query entirely inside one split.

T2Ranking answers one query with several passages, so a single query occupies
~14 rows. Splitting by row index scattered those rows and 89.4% of the test
queries had also been seen during training. The split now buckets on the query
text; this checks the property directly on the real config.

    python tools/verify_split.py --data /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flagscale.train.datasets.retrieval import split_accepts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
    )
    parser.add_argument("--modulo", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    files = sorted(Path(args.data).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data}")

    # group_key -> set of splits that group would land in
    per_query: dict[str, set[str]] = {}
    rows = {name: 0 for name in ("train", "validation", "test")}
    index = 0
    for file in files:
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(columns=["anchor"]):
            for anchor in batch.column(0).to_pylist():
                query = str(anchor or "")
                for split in ("train", "validation", "test"):
                    if split_accepts(index, split, args.modulo, group_key=query):
                        rows[split] += 1
                        per_query.setdefault(query, set()).add(split)
                index += 1
                if args.max_rows and index >= args.max_rows:
                    break
            if args.max_rows and index >= args.max_rows:
                break
        if args.max_rows and index >= args.max_rows:
            break

    leaked = {q: s for q, s in per_query.items() if len(s) > 1}
    print(f"rows scanned: {index}")
    print(f"unique queries: {len(per_query)}")
    for split, count in rows.items():
        print(f"  {split:10s} rows: {count}")
    print(f"queries spanning more than one split: {len(leaked)}")
    if leaked:
        example = next(iter(leaked.items()))
        print(f"  e.g. {example[0][:40]!r} -> {sorted(example[1])}")
        raise SystemExit("SPLIT LEAKS: a query must live in exactly one split")
    print("OK: every query belongs to exactly one split")


if __name__ == "__main__":
    main()
