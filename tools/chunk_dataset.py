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
    parser.add_argument(
        "--out",
        default=None,
        help="destination directory (required unless --dry-run)",
    )
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

    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is given")

    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted(Path(args.data).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.data}")

    out_dir = Path(args.out) if args.out else None
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    schema = pq.ParquetFile(files[0]).schema_arrow
    columns = schema.names

    # Pass 1: count train rows across the WHOLE corpus, so chunking can be even.
    # An earlier version stopped scanning as soon as it had written
    # max_chunks files, which took every row from the front of the corpus and
    # gave each chunk a different slice of the distribution. T2Ranking is ordered
    # by query, so that bias is real: late-file queries would never appear.
    total_seen = 0
    total_train = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["anchor"], batch_size=2048):
            for row in batch.to_pylist():
                total_seen += 1
                if split_accepts(
                    0, "train", args.modulo, group_key=str(row.get("anchor") or "")
                ):
                    total_train += 1

    if total_train == 0:
        raise SystemExit(f"the train split is empty under {args.data}")

    # Round-robin: train row i goes to chunk (i % n_chunks). Each chunk therefore
    # gets an interleaved sample of the WHOLE corpus rather than one contiguous
    # region, so no chunk is biased toward early or late queries. Deterministic,
    # and every train row lands in exactly one chunk (up to the per-chunk cap).
    n_chunks = min(args.max_chunks, max(1, total_train // args.rows_per_chunk))
    chunk_rows = [[] for _ in range(n_chunks)]
    accepted = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=columns, batch_size=1024):
            for row in batch.to_pylist():
                if not split_accepts(
                    0, "train", args.modulo, group_key=str(row.get("anchor") or "")
                ):
                    continue
                target = accepted % n_chunks
                if len(chunk_rows[target]) < args.rows_per_chunk:
                    chunk_rows[target].append(row)
                accepted += 1

    written: list[dict[str, object]] = []
    for idx, rows in enumerate(chunk_rows):
        if not rows:
            continue
        name = f"chunk-{idx:03d}.parquet"
        if not args.dry_run:
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, out_dir / name)
        written.append(
            {
                "file": name,
                "rows": len(rows),
                "rank_rows": len(rows) // args.world_size,
                "steps": len(rows) // (args.world_size * args.batch_size),
            }
        )

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
