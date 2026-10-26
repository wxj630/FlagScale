"""Audit the retrieval pipeline end to end before spending GPU-days on reruns.

Each check targets a mistake that would invalidate a multi-day run. They are
cheap: no model, no full epoch.

    python tools/audit_retrieval.py --data <triplet dir> [--clip-data <dir>]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def check_scheduler() -> None:
    """A streaming run must not decay to min_lr before the length is known."""

    from omegaconf import OmegaConf

    from flagscale.train.train_retrieval import build_lr_scheduler

    cfg = OmegaConf.create(
        {"lr": 2e-5, "min_lr": 2e-6, "lr_decay_style": "cosine", "warmup_steps": 200}
    )
    import torch

    param = torch.nn.Parameter(torch.zeros(1))
    optim = torch.optim.AdamW([param], lr=2e-5)

    # Unknown total, as every retrieval config is (no lr_decay_steps).
    sched, warmup, state = build_lr_scheduler(cfg, optim, total_steps=None)
    check(
        "scheduler: warmup preserved when total unknown",
        warmup == 200,
        f"warmup={warmup} (config asks 200)",
    )
    check("scheduler: total marked unknown", state["known"] is False)

    # Hold at base_lr until anchored.
    for _ in range(500):
        sched.step()
    lr_unanchored = optim.param_groups[0]["lr"]
    check(
        "scheduler: stays at base_lr before anchoring",
        abs(lr_unanchored - 2e-5) < 1e-9,
        f"lr={lr_unanchored:.3e} after 500 steps",
    )

    # Anchor to a 15000-step run and confirm it decays.
    state["total_steps"] = 15000
    state["known"] = True
    for _ in range(7500):
        sched.step()
    lr_mid = optim.param_groups[0]["lr"]
    check(
        "scheduler: decays after anchoring",
        2e-6 < lr_mid < 1.5e-5,
        f"lr={lr_mid:.3e} at step 8000 of 15000",
    )


def check_eval_defaults() -> None:
    """--max-samples must mean the whole split when omitted."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ev", Path(__file__).resolve().parent.parent / "examples/retrieval/evaluate_retrieval.py"
    )
    module = importlib.util.module_from_spec(spec)
    # The parser is built inside main(); re-create it by importing the module's
    # parser construction is not exposed, so inspect the default directly.
    src = (Path(__file__).resolve().parent.parent / "examples/retrieval/evaluate_retrieval.py").read_text()
    check(
        "eval: --max-samples defaults to whole split",
        'parser.add_argument("--max-samples", type=int, default=None)' in src,
        "a numeric default would cap every evaluation",
    )
    del module, spec


def check_split(data: str) -> None:
    """No query may appear in two splits."""

    from flagscale.train.datasets.retrieval import split_accepts

    import pyarrow.parquet as pq

    seen: dict[str, set[str]] = {}
    rows = 0
    for file in sorted(Path(data).glob("*.parquet")):
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(columns=["anchor"]):
            for anchor in batch.column(0).to_pylist():
                query = str(anchor or "")
                for split in ("train", "validation", "test"):
                    if split_accepts(rows, split, 20, group_key=query):
                        seen.setdefault(query, set()).add(split)
                rows += 1
    leaked = sum(1 for s in seen.values() if len(s) > 1)
    check(
        "split: every query lives in exactly one split",
        leaked == 0,
        f"{rows} rows, {len(seen)} queries, {leaked} leaked",
    )
    counts = {name: sum(1 for v in seen.values() if v == {name}) for name in
              ("train", "validation", "test")}
    check(
        "split: test set is non-trivial",
        counts["test"] > 1000,
        f"queries per split: {counts}",
    )


def check_cache_has_splits(cache_dir: str) -> None:
    """The Megatron token cache must carry per-row splits."""

    index_files = sorted(Path(cache_dir).glob("index-*.json")) if Path(cache_dir).is_dir() else []
    if not index_files:
        print(f"[SKIP] cache: no index under {cache_dir}")
        return
    import json

    missing = []
    for path in index_files:
        payload = json.loads(path.read_text())
        if "splits" not in payload:
            missing.append(path.name)
    check(
        "cache: every index carries per-row splits",
        not missing,
        f"{len(index_files)} index files, missing: {missing}",
    )


def check_config_semantics() -> None:
    """Batch/lr/epochs must match what the reports assume."""

    from omegaconf import OmegaConf

    conf_dir = Path(__file__).resolve().parent.parent / "examples/retrieval/conf/train"
    required = {"batch_size", "epochs"}
    problems = []
    for name in ("bge_m3", "clip_vit_large", "qwen3_reranker"):
        cfg = OmegaConf.load(conf_dir / f"{name}.yaml")
        miss = required - set(cfg.keys())
        if miss:
            problems.append(f"{name}: missing {miss}")
        opt = cfg.get("optimizer", {})
        for key in ("lr", "min_lr", "lr_decay_style"):
            if key not in opt:
                problems.append(f"{name}: optimizer.{key} missing")
        model = cfg.get("model", {})
        if "max_length" not in model:
            problems.append(f"{name}: model.max_length missing")
    check("config: all semantic fields present", not problems, "; ".join(problems))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", default="/flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15"
    )
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    check_scheduler()
    check_eval_defaults()
    check_config_semantics()
    check_split(args.data)
    if args.cache_dir:
        check_cache_has_splits(args.cache_dir)

    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED: {FAILS}")
        raise SystemExit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
