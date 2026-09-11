"""One-batch Torch-FL smoke test for the native BGE adapter."""

import torch_fl  # noqa: F401  # must precede torch
import torch
from omegaconf import OmegaConf

from flagscale.models.retrieval import build_retrieval_model
from flagscale.train.datasets.retrieval import build_split_loaders
from flagscale.train.losses.retrieval import loss_for_batch


def main() -> None:
    device = torch.device("flagos:0")
    model_cfg = OmegaConf.create(
        {
            "model_type": "bge_m3",
            "model_path": "/flagos-search-models/bge-m3/bge-m3",
            "max_length": 128,
            "torch_dtype": "bf16",
            "freeze_backbone": False,
        }
    )
    print("loading model", flush=True)
    model, processor = build_retrieval_model("bge_m3", model_cfg, device)
    print(f"loaded parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    data_cfg = OmegaConf.create(
        {
            "path": "/flagos-search-datasets/t2ranking/t2ranking/triplet-15",
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "num_negatives": 1,
            "split_modulo": 20,
            "max_samples": 2,
            "eval_max_samples": 2,
        }
    )
    loader, _, _ = build_split_loaders("embedding", data_cfg, 2, pin_memory=True)
    batch = next(iter(loader))
    print(f"batch keys={sorted(batch)}", flush=True)
    loss = loss_for_batch(model, processor, "embedding", batch, device, 0.05)
    print(f"loss={float(loss.detach().cpu()):.6f}", flush=True)
    loss.backward()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    optimizer.step()
    print("backward and AdamW step passed", flush=True)


if __name__ == "__main__":
    main()
