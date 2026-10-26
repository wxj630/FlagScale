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
    """Rank of the positive (candidate 0) among all candidates, ties as 1 + count."""

    ranks = []
    for row in predictions:
        scores = row["scores"]
        positive = scores[0]
        ranks.append(1 + sum(1 for s in scores if s > positive))
    return ranks


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
    preds = payload["predictions"]
    ranks = ranks_from_scores(preds)
    n = len(ranks)
    if n == 0:
        raise SystemExit("no predictions in file")

    if args.verify_rank:
        stored = [p["rank"] for p in preds]
        mismatches = sum(1 for a, b in zip(stored, ranks) if a != b)
        print(f"verify: {mismatches} / {n} rows disagree with stored rank")
        if mismatches:
            first = next(i for i, (a, b) in enumerate(zip(stored, ranks)) if a != b)
            print(f"  first at row {first}: stored={stored[first]} from-scores={ranks[first]}")

    label = f"{payload.get('model_type', '?')} [{payload.get('split', '?')}] {args.predictions.name}"
    print(f"{label}: {n} queries, {payload.get('num_negatives', '?')} negatives")

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

    here = summarize(ranks)
    for key, value in here.items():
        print(f"  {key:12s} {value:.4f}")

    if args.compare:
        other_payload = load(args.compare)
        other = summarize(ranks_from_scores(other_payload["predictions"]))
        print(f"\n  delta vs {args.compare.name} (this minus that)")
        for key in here:
            if key in other:
                print(f"  {key:12s} {here[key] - other[key]:+.4f}")


if __name__ == "__main__":
    main()
