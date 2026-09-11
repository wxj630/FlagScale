"""Retrieval quality evaluation for the native retrieval adapters.

The training loop only reports contrastive/ranking loss, which says little
about downstream retrieval quality.  This script loads a trained (or base)
checkpoint and reports rank-based metrics on the provided data:

* ``embedding`` / ``reranker``: for every query, rank its positive passage
  against that query's hard negatives.  Reported as Recall@k and MRR.
* ``clip`` / ``vl_embedding``: in-batch image/text retrieval with symmetric
  ranking, reported as Recall@k and MRR in both directions.

It reuses the same adapters and dataset code as training, so it never has to
reimplement preprocessing.  Run it on CPU or on a single accelerator; it does
not require torchrun.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# torch-fl must claim the PrivateUse1 slot before the first flagos tensor.
if os.environ.get("FLAGSCALE_DEVICE", "").strip().lower() in {"flagos", "privateuseone"}:
    import torch_fl  # noqa: F401

import torch
from omegaconf import OmegaConf

from flagscale.models.retrieval import build_retrieval_model
from flagscale.train.datasets.retrieval import build_dataset
from flagscale.train.losses.retrieval import embedding_batch, reranker_batch


def rank_metrics(ranks: list[int], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    if not ranks:
        raise RuntimeError("No queries were evaluated")
    total = len(ranks)
    metrics: dict[str, float] = {"count": total}
    for k in ks:
        metrics[f"recall@{k}"] = sum(1 for rank in ranks if rank <= k) / total
    metrics["mrr"] = sum(1.0 / rank for rank in ranks) / total
    metrics["mean_rank"] = sum(ranks) / total
    return metrics


def _batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_ranked(model, rows: list[dict], task: str, device, batch_size: int, negative_keys: list[str]):
    """Rank each query's positive against its hard negatives."""

    ranks: list[int] = []
    for chunk in _batched(rows, batch_size):
        anchors = [row["anchor"] for row in chunk]
        if task == "reranker":
            pair_lists: list[list[str]] = []
            groups: list[int] = []
            for row in chunk:
                docs = [row["positive"]] + [row[key] for key in negative_keys]
                for doc in docs:
                    pair_lists.append([row["anchor"], doc])
                groups.append(len(docs))
            scores = reranker_batch(model, pair_lists, device)
            offset = 0
            for size in groups:
                block = scores[offset : offset + size]
                offset += size
                ranks.append(1 + int((block > block[0]).sum().item()))
        else:
            anchor_emb = embedding_batch(model, anchors, device)
            candidate_blocks = []
            for key in ["positive", *negative_keys]:
                candidate_blocks.append(embedding_batch(model, [row[key] for row in chunk], device))
            candidates = torch.stack(candidate_blocks, dim=1)  # (B, 1+K, D)
            scores = torch.einsum("bd,bkd->bk", anchor_emb, candidates)
            positive = scores[:, :1]
            ranks.extend((1 + (scores > positive).sum(dim=1)).tolist())
    return ranks


@torch.no_grad()
def evaluate_in_batch(model, rows: list[dict], device, batch_size: int, ks: tuple[int, ...]):
    """Symmetric in-batch image/text retrieval (CLIP / VL embedding)."""

    text_ranks: list[int] = []
    image_ranks: list[int] = []
    for chunk in _batched(rows, batch_size):
        text_inputs = [{"text": row["text"]} for row in chunk]
        image_inputs = [{"image": row["image"]} for row in chunk]
        text_emb = embedding_batch(model, text_inputs, device)
        image_emb = embedding_batch(model, image_inputs, device)
        sim = text_emb @ image_emb.T
        labels = torch.arange(sim.shape[0], device=sim.device)
        text_ranks.extend((1 + (sim > sim[labels, labels].unsqueeze(1)).sum(dim=1)).tolist())
        sim_t = sim.T
        image_ranks.extend((1 + (sim_t > sim_t[labels, labels].unsqueeze(1)).sum(dim=1)).tolist())
    return {
        "text_to_image": rank_metrics(text_ranks, ks),
        "image_to_text": rank_metrics(image_ranks, ks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--checkpoint", default=None, help="trained checkpoint dir; falls back to --model-path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--task", required=True, choices=["embedding", "reranker", "clip", "vl_embedding"])
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--split-modulo", type=int, default=20)
    parser.add_argument("--num-negatives", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.05, help="unused for ranking; kept for clarity")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="write metrics JSON here")
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device.startswith("flagos"):
        torch.flagos.set_device(0)

    model_path = args.checkpoint or args.model_path
    model_cfg = OmegaConf.create(
        {
            "model_type": args.model_type,
            "model_path": model_path,
            "max_length": args.max_length,
            "torch_dtype": "bf16",
            "freeze_backbone": False,
            "use_score_head": args.task == "reranker",
            "projection_dim": 0 if args.task != "vl_embedding" else 0,
        }
    )
    model, _ = build_retrieval_model(args.model_type, model_cfg, device)
    model.eval()

    data_cfg = OmegaConf.create(
        {
            "path": args.data_path,
            "split_modulo": args.split_modulo,
            "num_negatives": args.num_negatives,
            "shuffle_buffer": 0,
        }
    )
    dataset = build_dataset(args.task, data_cfg, args.data_path, args.split, args.max_samples)
    rows = list(dataset)
    if not rows:
        raise RuntimeError(f"No rows for split {args.split!r} in {args.data_path}")

    if args.task in {"clip", "vl_embedding"}:
        metrics = evaluate_in_batch(model, rows, device, args.batch_size, (1, 5, 10))
    else:
        negative_keys = [f"negative_{i}" for i in range(1, args.num_negatives + 1)
                         if f"negative_{i}" in rows[0]]
        ranks = evaluate_ranked(model, rows, args.task, device, args.batch_size, negative_keys)
        metrics = rank_metrics(ranks, (1, 5, 10))
        metrics["num_negatives"] = len(negative_keys)

    result = {
        "model_type": args.model_type,
        "task": args.task,
        "checkpoint": model_path,
        "split": args.split,
        "samples": len(rows),
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
