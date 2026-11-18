"""Cut the T2Ranking train split into disjoint parquet chunks.

Why this exists: P800 training dies inside the vendor XCCL stack at a step that
varies between runs (2,260 observed three times, then 1,220 on 2026-11-18), and
nothing survives a death except what is already on disk. The workaround is to
run training as many short jobs, each reading its own slice of the corpus.

Two ways to give a job its own slice:

* ``data.skip_samples`` -- the reader consumes and discards N rows first. This
  works but is fragile: the skip has to be applied after the shuffle buffer (it
  was applied before it at first, which re-read up to 20k rows of the previous
  chunk), and a skip past the end of the stream yields nothing rather than
  wrapping. Both mistakes already cost real runs.
* physical chunks -- each job reads a different file. Nothing to get wrong, a
  chunk cannot leak into another one, and ``ls`` shows exactly what each job saw.

This produces the second. Splitting uses ``split_accepts`` (imported, not
reimplemented) so the chunks contain only rows the training split would have
taken; validation and test rows are excluded by construction, which is what
keeps the reported nDCG honest.

    python tools/chunk_dataset.py \
        --data /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15 \
        --out  /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15-chunks \
        --rows-per-chunk 28800 --max-chunks 10

Each output file is ``chunk-000.tsv.parquet``-style: ``chunk-000.parquet``,
``chunk-001.parquet``, ... plus a ``manifest.json`` recording the row counts and
the exact filter that produced them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flagscale.train.datasets.retrieval import split_accepts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15",
        help="source directory of train-*.parquet shards",
    )
    parser.add_argument("--out", required=True, help="destination directory")
    parser.add_argument(
        "--rows-per-chunk",
        type=int,
        default=28800,
        help="rows per chunk across ALL ranks (2 ranks x 16 rows x 900 steps = 28800)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=10, help="stop after this many chunks"
    )
    parser.add_argument("--modulo", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="only used to report steps-per-chunk in the manifest",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="only used to report steps-per-chunk in the manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count rows per chunk without writing files",
    )
    args = parser.parse_args()

    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(args.data).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data}")

    out_dir = Path(args.out)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    schema = pq.ParquetFile(files[0]).schema_arrow
    columns = schema.names

    chunk_index = 0
    chunk_rows: list[dict[str, str]] = []
    written: list[dict[str, object]] = []
    total_seen = 0
    total_train = 0

    def flush() -> None:
        nonlocal chunk_index, chunk_rows
        if not chunk_rows:
            return
        name = f"chunk-{chunk_index:03d}.parquet"
        if not args.dry_run:
            table = pa.Table.from_pylist(chunk_rows, schema=schema)
            pq.write_table(table, out_dir / name)
        written.append(
            {
                "file": name,
                "rows": len(chunk_rows),
                "rank_rows": len(chunk_rows) // args.world_size,
                "steps": len(chunk_rows) // (args.world_size * args.batch_size),
            }
        )
        chunk_index += 1
        chunk_rows = []

    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=1024):
            for row in batch.to_pylist():
                total_seen += 1
                # Same predicate the trainer applies, on the same key: bucketing
                # on `anchor` keeps every row of a query in one split.
                if not split_accepts(
                    0,
                    "train",
                    args.modulo,
                    group_key=str(row.get("anchor") or ""),
                ):
                    continue
                total_train += 1
                chunk_rows.append(row)
                if len(chunk_rows) >= args.rows_per_chunk:
                    flush()
                    if len(written) >= args.max_chunks:
                        break
            if len(written) >= args.max_chunks:
                break
        if len(written) >= args.max_chunks:
            break

    if len(written) < args.max_chunks:
        flush()

    manifest = {
        "source": str(Path(args.data).resolve()),
        "modulo": args.modulo,
        "split": "train",
        "rows_per_chunk": args.rows_per_chunk,
        "batch_size": args.batch_size,
        "world_size": args.world_size,
        "source_files": [p.name for p in files],
        "rows_scanned": total_seen,
        "rows_in_train_split": total_train,
        "rows_written": sum(c["rows"] for c in written),
        "chunks": written,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    print(f"scanned {total_seen} rows, train split holds {total_train}")
    print(f"wrote {len(written)} chunks x {args.rows_per_chunk} rows")
    for c in written:
        print(f"  {c['file']}: {c['rows']} rows -> {c['steps']} steps/rank")


if __name__ == "__main__":
    main()
