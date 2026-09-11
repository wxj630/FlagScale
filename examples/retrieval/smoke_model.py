"""One-batch smoke test for a native FlagScale retrieval adapter."""

from __future__ import annotations

import argparse

import torch_fl  # noqa: F401  # must precede torch
import torch
from omegaconf import OmegaConf

from flagscale.models.retrieval import build_retrieval_model
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch


_MODELS = {
    "qwen3_embedding": {
        "task": "embedding",
        "path": "/flagos-search-models/Qwen3-Embedding-0.6B",
        "data": "/flagos-search-datasets/t2ranking/t2ranking/triplet-15",
        "max_length": 128,
        "num_negatives": 1,
    },
    "clip_vit_large": {
        "task": "clip",
        "path": "/flagos-search-models/clip-vit-large-patch14/clip-vit-large-patch14",
        "data": "/flagos-search-datasets/wds_mscoco_captions2017/wds_mscoco_captions2017/train",
        "max_length": 77,
        "num_negatives": 1,
    },
    "qwen3_reranker": {
        "task": "reranker",
        "path": "/flagos-search-models/Qwen3-Reranker-8B/Qwen3-Reranker-8B",
        "data": "/flagos-search-datasets/t2ranking/t2ranking/triplet-15",
        "max_length": 128,
        "num_negatives": 1,
        "freeze_backbone": True,
        "use_score_head": True,
    },
    "qwen3_vl_embedding": {
        "task": "vl_embedding",
        "path": "/flagos-search-models/Qwen3-VL-Embedding-8B/Qwen3-VL-Embedding-8B",
        "data": "/flagos-search-datasets/wds_mscoco_captions2017/wds_mscoco_captions2017/train",
        # Image placeholders expand to a few hundred tokens; do not truncate
        # them in the processor smoke test.
        "max_length": 2048,
        "num_negatives": 1,
        "freeze_backbone": True,
        "projection_dim": 1024,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(_MODELS))
    args = parser.parse_args()
    spec = _MODELS[args.model]
    device = torch.device("flagos:0")
    model_cfg = OmegaConf.create(
        {
            "model_type": args.model,
            "model_path": spec["path"],
            "max_length": spec["max_length"],
            "torch_dtype": "bf16",
            "freeze_backbone": bool(spec.get("freeze_backbone", False)),
            "use_score_head": bool(spec.get("use_score_head", False)),
            "projection_dim": int(spec.get("projection_dim", 0)),
        }
    )
    print(f"loading {args.model}", flush=True)
    model, processor = build_retrieval_model(args.model, model_cfg, device)
    print(f"loaded parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    data_cfg = OmegaConf.create(
        {
            "path": spec["data"],
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "num_negatives": spec["num_negatives"],
            "split_modulo": 20,
            "max_samples": 2,
            "eval_max_samples": 2,
        }
    )
    loader, _, _ = build_split_loaders(spec["task"], data_cfg, 2, pin_memory=True)
    batch = next(iter(loader))
    print(f"batch keys={sorted(batch)}", flush=True)
    loss = loss_for_batch(model, processor, spec["task"], batch, device, 0.05)
    print(f"loss={float(loss.detach().cpu()):.6f}", flush=True)
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.step()
    print("backward and AdamW step passed", flush=True)


if __name__ == "__main__":
    main()
