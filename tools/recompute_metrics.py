"""Recompute retrieval metrics from a saved predictions file.

`evaluate_retrieval.py --save-predictions` writes one row per query with the
full candidate score vector, so a new metric or a different cutoff can be
computed without re-running the model. That matters because a full 8B pass over
17k queries takes hours.

    python tools/recompute_metrics.py predictions.json
    python tools/recompute_metrics.py predictions.json --cutoffs 1 3 5 10 20 --ndcg
    python tools/recompute_metrics.py predictions.json --by-rank --compare base.predictions.json

The rank in the file was produced with the metrics the run used; this script
recomputes recall/MRR from `scores` rather than trusting `rank`, so a scoring bug
in the original run would show up as a disagreement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if "predictions" not in payload:
        raise SystemExit(f"{path} has no 'predictions' key; was it written by --save-predictions?")
    return payload


def ranks_from_scores(predictions: list[dict]) -> list[int]:
    """Rank of the ground-truth candidate, ties counted as 1 + strictly-greater.

    Two shapes are accepted, because image-text models store one row per item
    over the whole candidate pool while text models store one row per query with
    the positive first:

    * ``ground_truth`` is an index into ``scores`` (any value)
    * rows without it default to index 0, the positive-first layout
    """

    ranks = []
    for row in predictions:
        scores = row["scores"]
        gt = int(row.get("ground_truth", 0))
        if not 0 <= gt < len(scores):
            raise SystemExit(
                f"ground_truth {gt} out of range for a row with {len(scores)} scores"
            )
        positive = scores[gt]
        ranks.append(1 + sum(1 for s in scores if s > positive))
    return ranks


def all_rank_sets(payload: dict) -> dict[str, list[int]]:
    """Every rank list a predictions file describes, keyed by direction.

    Text models have one; image-text models have two (text->image and the
    reverse), which are reported separately and must not be mixed.
    """

    preds = payload["predictions"]
    if isinstance(preds, dict):
        return {direction: ranks_from_scores(rows) for direction, rows in preds.items()}
    return {"": ranks_from_scores(preds)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[1, 3, 5, 10, 20])
    parser.add_argument("--ndcg", action="store_true", help="also report nDCG@k")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="another predictions file; prints a per-cutoff delta (this minus that)",
    )
    parser.add_argument(
        "--verify-rank",
        action="store_true",
        help="assert the stored 'rank' matches the score-derived rank",
    )
    args = parser.parse_args()

    payload = load(args.predictions)
    rank_sets = all_rank_sets(payload)
    if not rank_sets or all(len(v) == 0 for v in rank_sets.values()):
        raise SystemExit("no predictions in file")

    if args.verify_rank:
        # Text models store `rank`; image-text rows do not, so only check those
        # that have it.
        preds = payload["predictions"]
        flat = preds if isinstance(preds, list) else [
            row for rows in preds.values() for row in rows
        ]
        if any("rank" in row for row in flat):
            stored = [row["rank"] for row in flat if "rank" in row]
            derived = ranks_from_scores([row for row in flat if "rank" in row])
            mismatches = sum(1 for a, b in zip(stored, derived) if a != b)
            print(f"verify: {mismatches} / {len(stored)} rows disagree with stored rank")
        else:
            print("verify: no stored rank column in this file (image-text layout)")

    label = f"{payload.get('model_type', '?')} [{payload.get('split', '?')}] {args.predictions.name}"
    print(f"{label}: {payload.get('num_negatives', '?')} negatives, "
          f"directions={list(rank_sets)}")

    def summarize(rs: list[int]) -> dict[str, float]:
        out = {}
        for k in args.cutoffs:
            out[f"recall@{k}"] = sum(1 for r in rs if r <= k) / len(rs)
        out["mrr"] = sum(1.0 / r for r in rs) / len(rs)
        out["mean_rank"] = sum(rs) / len(rs)
        if args.ndcg:
            for k in args.cutoffs:
                out[f"ndcg@{k}"] = sum(
                    1.0 / math.log2(r + 1) if r <= k else 0.0 for r in rs
                ) / len(rs)
        return out

    other = None
    if args.compare:
        other = {
            direction: summarize(rs)
            for direction, rs in all_rank_sets(load(args.compare)).items()
        }

    for direction, ranks in rank_sets.items():
        if direction:
            print(f"\n  [{direction}] {len(ranks)} items")
        here = summarize(ranks)
        for key, value in here.items():
            print(f"  {key:12s} {value:.4f}")
        if other is not None and direction in other:
            print(f"  delta vs {args.compare.name} (this minus that)")
            for key in here:
                if key in other[direction]:
                    print(f"  {key:12s} {here[key] - other[direction][key]:+.4f}")


if __name__ == "__main__":
    main()
