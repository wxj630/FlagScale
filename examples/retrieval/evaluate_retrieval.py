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
import sys
from pathlib import Path

# torch-fl must claim the PrivateUse1 slot before the first flagos tensor is
# created.  Check both the environment and the requested device so that
# ``--device flagos`` works on its own, without relying on the caller to export
# FLAGSCALE_DEVICE.
_ENV_DEVICE = os.environ.get("FLAGSCALE_DEVICE", "").strip().lower()
_CLI_DEVICE = next(
    (
        arg.split("=", 1)[1]
        for arg in sys.argv[1:]
        if arg.startswith("--device=")
    ),
    None,
)
if _CLI_DEVICE is None and "--device" in sys.argv:
    index = sys.argv.index("--device")
    _CLI_DEVICE = sys.argv[index + 1] if index + 1 < len(sys.argv) else ""
_CLI_DEVICE = (_CLI_DEVICE or "").strip().lower()
if _ENV_DEVICE in {"flagos", "privateuseone"} or _CLI_DEVICE.startswith("flagos"):
    import torch_fl  # noqa: F401

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from flagscale.models.retrieval import build_retrieval_model
from flagscale.train.datasets.retrieval import build_dataset
from flagscale.train.losses.retrieval import embedding_batch, move_features, reranker_batch


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


@torch.no_grad()
def evaluate_ranked(model, rows: list[dict], task: str, device, batch_size: int, negative_keys: list[str]):
    """Rank each query's positive against its hard negatives."""

    ranks: list[int] = []
    total = len(rows)
    for index, chunk in enumerate(_batched(rows, batch_size)):
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
        # Full test splits take hours on the 8B models, so report progress
        # instead of leaving the caller blind until the final JSON.
        if (index + 1) % 50 == 0:
            done = len(ranks)
            print(
                f"  [progress] {done}/{total} queries "
                f"({done / total * 100:.1f}%) running_mrr={_mrr(ranks):.4f}",
                flush=True,
            )
    return ranks


def _mrr(ranks: list[int]) -> float:
    return sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0


@torch.no_grad()
def _clip_features(backbone, kind: str, inputs: dict) -> torch.Tensor:
    """Text or image features as a tensor, across Transformers versions.

    Transformers 4.x returns a bare tensor from ``get_text_features`` /
    ``get_image_features``; 5.x returns a ``BaseModelOutputWithPooling``.
    Accept both instead of pinning a version.
    """

    getter = backbone.get_text_features if kind == "text" else backbone.get_image_features
    features = getter(**inputs)
    if torch.is_tensor(features):
        return features
    for attribute in ("pooler_output", f"{kind}_embeds"):
        candidate = getattr(features, attribute, None)
        if torch.is_tensor(candidate):
            return candidate
    raise RuntimeError(
        f"{type(features).__name__} from get_{kind}_features has no usable "
        f"embedding field: {list(features)}"
    )


def _clip_embeddings(model, chunk: list[dict], device) -> tuple[torch.Tensor, torch.Tensor]:
    """Text and image features for a CLIP batch.

    CLIP's own forward needs text and images together, so it cannot go through
    the generic ``prepare_embedding_inputs`` path; Transformers exposes separate
    feature extractors which is exactly what retrieval ranking needs.
    """

    base = model.module if hasattr(model, "module") else model
    processor = base.processor
    backbone = base.backbone
    texts = [row["text"] for row in chunk]
    images = [row["image"] for row in chunk]
    text_inputs = move_features(processor(text=texts, return_tensors="pt", padding=True, truncation=True), device)
    image_inputs = move_features(processor(images=images, return_tensors="pt"), device)
    text_emb = F.normalize(_clip_features(backbone, "text", text_inputs).float(), p=2, dim=-1)
    image_emb = F.normalize(_clip_features(backbone, "image", image_inputs).float(), p=2, dim=-1)
    return text_emb, image_emb


@torch.no_grad()
def evaluate_in_batch(model, rows: list[dict], device, batch_size: int, ks: tuple[int, ...], task: str = ""):
    """Symmetric image/text retrieval over the **whole** evaluation set.

    The name is historical: an earlier version scored each batch in isolation,
    which made the candidate pool the batch size (64) instead of the full test
    set (5000). That inflates recall by two orders of magnitude -- the random
    baseline moves from 1/5000 to 1/64 -- and makes results depend on the batch
    size, so they could not be compared with published MSCOCO numbers.

    The protocol now matches the standard CLIP evaluation: encode every image
    and every text once, build the full N x N similarity matrix, and rank each
    text against all images (and vice versa). Memory for N=5000 is ~100 MB per
    direction at fp32, so this is affordable.
    """

    total = len(rows)
    text_embeddings: list[torch.Tensor] = []
    image_embeddings: list[torch.Tensor] = []
    for index, chunk in enumerate(_batched(rows, batch_size)):
        if task == "clip":
            text_emb, image_emb = _clip_embeddings(model, chunk, device)
        else:
            text_inputs = [{"text": row["text"]} for row in chunk]
            image_inputs = [{"image": row["image"]} for row in chunk]
            text_emb = embedding_batch(model, text_inputs, device)
            image_emb = embedding_batch(model, image_inputs, device)
        text_embeddings.append(text_emb.float().cpu())
        image_embeddings.append(image_emb.float().cpu())
        if (index + 1) % 20 == 0:
            done = min((index + 1) * batch_size, total)
            print(f"  [encode] {done}/{total} ({done / total * 100:.1f}%)", flush=True)

    text_all = torch.cat(text_embeddings, dim=0)      # (N, D)
    image_all = torch.cat(image_embeddings, dim=0)    # (N, D)
    print(f"  [encode] done: {text_all.shape[0]} texts x {image_all.shape[0]} images", flush=True)

    # Full N x N similarity, computed on the accelerator in row chunks so the
    # matrix itself does not have to live there in one piece.
    text_ranks, image_ranks = _full_retrieval_ranks(text_all, image_all, device)
    return {
        "text_to_image": rank_metrics(text_ranks, ks),
        "image_to_text": rank_metrics(image_ranks, ks),
    }


@torch.no_grad()
def evaluate_in_batch_streaming(
    model,
    dataset,
    device,
    batch_size: int,
    ks: tuple[int, ...],
    task: str = "",
    row_chunk: int = 1024,
) -> dict:
    """Full-set image/text retrieval, encoding straight from the stream.

    Same protocol as :func:`evaluate_in_batch` (rank every text against every
    image of the evaluation set) but it never holds more than one batch of
    decoded images in host memory -- materialising 5000 PIL images costs
    5-15 GB and is what got the process OOM-killed on the P800 host.
    """

    import itertools

    text_embeddings: list[torch.Tensor] = []
    image_embeddings: list[torch.Tensor] = []
    count = 0
    stream = iter(dataset)
    while True:
        chunk = list(itertools.islice(stream, batch_size))
        if not chunk:
            break
        if task == "clip":
            text_emb, image_emb = _clip_embeddings(model, chunk, device)
        else:
            text_inputs = [{"text": row["text"]} for row in chunk]
            image_inputs = [{"image": row["image"]} for row in chunk]
            text_emb = embedding_batch(model, text_inputs, device)
            image_emb = embedding_batch(model, image_inputs, device)
        text_embeddings.append(text_emb.float().cpu())
        image_embeddings.append(image_emb.float().cpu())
        count += len(chunk)
        del chunk
        if count % (batch_size * 10) == 0:
            print(f"  [encode] {count} rows", flush=True)

    if not text_embeddings:
        raise RuntimeError("Evaluation dataset produced no rows")
    text_all = torch.cat(text_embeddings, dim=0)
    image_all = torch.cat(image_embeddings, dim=0)
    print(f"  [encode] done: {text_all.shape[0]} texts x {image_all.shape[0]} images", flush=True)

    text_ranks, image_ranks = _full_retrieval_ranks(text_all, image_all, device, row_chunk)
    result = {
        "text_to_image": rank_metrics(text_ranks, ks),
        "image_to_text": rank_metrics(image_ranks, ks),
        "samples": int(text_all.shape[0]),
    }
    return result


def _full_retrieval_ranks(
    text_all: torch.Tensor,
    image_all: torch.Tensor,
    device: torch.device,
    row_chunk: int = 1024,
) -> tuple[list[int], list[int]]:
    """Ranks for text->image and image->text against the full candidate pool.

    Row-chunked so peak device memory is ``row_chunk * N`` floats (~20 MB for
    N=5000, chunk=1024) rather than the whole matrix.
    """

    n = text_all.shape[0]
    # The diagonal is the paired caption/image; the rank counts how many other
    # candidates score strictly higher, so ties do not inflate the rank.
    text_ranks: list[int] = []
    for start in range(0, n, row_chunk):
        block = text_all[start : start + row_chunk].to(device)
        sim = block @ image_all.to(device).T                      # (chunk, N)
        paired = sim[torch.arange(block.shape[0], device=device), torch.arange(start, start + block.shape[0], device=device)]
        better = (sim > paired.unsqueeze(1)).sum(dim=1)
        text_ranks.extend((better + 1).tolist())
        del sim, paired, better

    image_ranks: list[int] = []
    for start in range(0, n, row_chunk):
        block = image_all[start : start + row_chunk].to(device)
        sim = block @ text_all.to(device).T
        paired = sim[torch.arange(block.shape[0], device=device), torch.arange(start, start + block.shape[0], device=device)]
        better = (sim > paired.unsqueeze(1)).sum(dim=1)
        image_ranks.extend((better + 1).tolist())
        del sim, paired, better

    return text_ranks, image_ranks


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
    parser.add_argument(
        "--train-config",
        default=None,
        help="training YAML for this model; its model.* settings (use_score_head, "
             "projection_dim, ...) are reused so evaluation matches training",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device.startswith("flagos"):
        torch.flagos.set_device(0)

    model_path = args.checkpoint or args.model_path
    # Take the model settings from the training config when one is given. A
    # previous run inferred use_score_head from the task name and silently
    # evaluated a trained reranker through a freshly initialised score head,
    # which made a correctly fine-tuned model look worse than random.
    if args.train_config:
        train_cfg = OmegaConf.load(args.train_config)
        model_cfg = OmegaConf.create(
            {
                "model_type": train_cfg.model.get("model_type", args.model_type),
                "model_path": model_path,
                "max_length": int(train_cfg.model.get("max_length", args.max_length)),
                "torch_dtype": str(train_cfg.model.get("torch_dtype", "bf16")),
                "freeze_backbone": bool(train_cfg.model.get("freeze_backbone", False)),
                "use_score_head": bool(train_cfg.model.get("use_score_head", False)),
                "projection_dim": int(train_cfg.model.get("projection_dim", 0)),
            }
        )
    else:
        model_cfg = OmegaConf.create(
            {
                "model_type": args.model_type,
                "model_path": model_path,
                "max_length": args.max_length,
                "torch_dtype": "bf16",
                "freeze_backbone": False,
                "use_score_head": False,
                "projection_dim": 0,
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

    if args.task in {"clip", "vl_embedding"}:
        # Stream instead of materialising every row: each row holds a decoded PIL
        # image (megabytes each), so list()-ing the 5000-image MSCOCO split costs
        # 5-15 GB of host RAM and gets the process OOM-killed on a busy host.
        # Only the embeddings need to persist.
        metrics = evaluate_in_batch_streaming(
            model, dataset, device, args.batch_size, (1, 5, 10), args.task
        )
        result = {
            "model_type": args.model_type,
            "task": args.task,
            "checkpoint": model_path,
            "split": args.split,
            "samples": metrics.pop("samples"),
            "metrics": metrics,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.output:
            with open(args.output, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=2)
        return

    rows = list(dataset)
    if not rows:
        raise RuntimeError(f"No rows for split {args.split!r} in {args.data_path}")

    negative_keys = [f"negative_{i}" for i in range(1, args.num_negatives + 1)
                     if f"negative_{i}" in rows[0]]
    # Rank by length before batching. Every query's candidate set stays in
    # one batch, so ordering by length cannot change any query's result, but
    # it stops a single long row from padding a whole batch to the max: with
    # random order each 192-sequence batch padded to 640 tokens, after
    # sorting the batch max tracks the local length instead.
    rows.sort(key=lambda row: len(row.get("anchor", "")) + len(row.get("positive", "")))
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
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
