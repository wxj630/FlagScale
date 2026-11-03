"""Estimate pretrain + SFT cost for one epoch on a single 8-GPU A100 server.

Answers a different question from ``docs/retrieval/06-gpu-hours.md`` §一-§八:
there the budget follows each paper's own recipe (CLIP 32 epochs, reranker
3 epochs); here every stage is walked exactly once.

Three data calibers, because the papers publish three different populations:

  A. paper-full  -- everything the paper claims to train on (upper bound)
  B. public      -- the subset actually downloadable today
  C. mean        -- B / number_of_datasets, i.e. one representative dataset

Throughput is per A100 and comes from ``docs/retrieval/07-a100.md`` except
where marked below. Update ``THROUGHPUT`` and ``STAGES`` when measurements land;
every number in the doc is reproducible with ``python estimate_gpu_hours.py``.

    python examples/retrieval/estimate_gpu_hours.py
    python examples/retrieval/estimate_gpu_hours.py --markdown
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# texts/s or pairs/s on one A100. Measured end to end unless noted.
THROUGHPUT = {
    # 07-a100.md: 23,010 steps / 39,017.5 s, 200 texts per step.
    "bge_512": 118.0,
    # Extrapolated from bge_512 by sequence length (512 -> 8192). NOT measured:
    # attention is O(L^2), so the true value can be 2x lower. Biggest uncertainty
    # in the whole estimate; unblock by running ~100 RetroMAE steps at 8192.
    "bge_8192": 7.4,
    # ASSUMED equal to bge_512: same ~0.6B dual encoder and sequence length.
    # Not measured. If the paper trained at 32k the real number is far lower.
    "qwen3_emb_512": 118.0,
    # 07-a100.md: 5.51 pairs/s across 2 cards, so 2.755 per card.
    "reranker_640": 5.51 / 2,
    # 07-a100.md: 318,720 samples / 2,679.1 s.
    "clip_77": 119.0,
}


@dataclass(frozen=True)
class Stage:
    model: str
    stage: str
    rate: str
    unit: str
    # Size under each caliber, already converted to the unit the rate is in
    # (a text pair costs two encoder passes, so pairs are doubled for bi-encoders).
    full: float
    public: float
    mean: float
    note: str = ""


# Divisors are spelled out in the note so the mean caliber can be re-checked.
STAGES = [
    Stage(
        "BGE-M3", "pretrain-A RetroMAE@8192", "bge_8192", "texts",
        full=184e6, public=184e6, mean=184e6 / 3,
        note="Pile/Wudao/mC4, all public -> A == B",
    ),
    Stage(
        "BGE-M3", "pretrain-B contrastive@512", "bge_512", "texts",
        full=1.2e9 * 2, public=928.3e6 * 2, mean=928.3e6 / 4 * 2,
        note="77% public (MTP 291.1M missing); 4 public tables",
    ),
    Stage(
        "BGE-M3", "SFT@512", "bge_512", "texts",
        full=1.648e6 * 9, public=1.5755e6 * 9, mean=1.5755e6 / 16 * 9,
        note="9 texts/row (1 query + 1 pos + 7 neg); 17 sets full, 16 public",
    ),
    Stage(
        "Qwen3-Emb-0.6B", "pretrain weak-sup@512", "qwen3_emb_512", "texts",
        full=150e6 * 2, public=0, mean=0,
        note="Qwen3-32B synthetic, unreleased",
    ),
    Stage(
        "Qwen3-Emb-0.6B", "SFT@512", "qwen3_emb_512", "texts",
        full=19e6 * 2, public=2.8e6 * 2, mean=2.8e6 / 12 * 2,
        note="7M labeled + 12M synthetic; public = 40% of the 7M over 12 sets",
    ),
    Stage(
        "Qwen3-Rerank-8B", "SFT@640", "reranker_640", "pairs",
        full=19e6, public=2.8e6, mean=2.8e6 / 12,
        note="same data as the embedding model, no pretrain stage",
    ),
    Stage(
        "CLIP ViT-L/14", "pretrain@77", "clip_77", "pairs",
        full=400e6, public=15e6, mean=15e6,
        note="WIT unreleased; YFCC filtered to 15M is the paper's own substitute",
    ),
    Stage(
        "CLIP ViT-L/14", "SFT@77", "clip_77", "pairs",
        full=0, public=106240, mean=106240,
        note="no such stage in the paper; our MSCOCO finetune",
    ),
]

CALIBERS = ("full", "public", "mean")


def gpu_hours(units: float, rate_per_s: float) -> float:
    return units / (rate_per_s * 3600.0)


def fmt(x: float) -> str:
    if x >= 1:
        return f"{x:,.0f}"
    return f"{x:.1f}" if x > 0 else "0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="emit markdown tables")
    ap.add_argument("--gpus", type=int, default=8, help="GPUs per server")
    args = ap.parse_args()

    totals = {c: 0.0 for c in CALIBERS}
    sft_totals = {c: 0.0 for c in CALIBERS}
    per_model: dict[str, dict[str, float]] = {}

    if args.markdown:
        print("| model | stage | A paper-full | B public | C mean |")
        print("|---|---|---|---|---|")

    for st in STAGES:
        hours = {c: gpu_hours(getattr(st, c), THROUGHPUT[st.rate]) for c in CALIBERS}
        m = per_model.setdefault(st.model, {c: 0.0 for c in CALIBERS})
        for c in CALIBERS:
            totals[c] += hours[c]
            m[c] += hours[c]
        if st.stage.startswith("SFT"):
            for c in CALIBERS:
                sft_totals[c] += hours[c]

        if args.markdown:
            print(
                f"| {st.model} | {st.stage} | "
                + " | ".join(fmt(hours[c]) for c in CALIBERS)
                + " |"
            )
        else:
            print(f"{st.model} / {st.stage}  [{st.note}]")
            for c in CALIBERS:
                print(f"    {c:<7} {getattr(st, c):>16,.0f} {st.unit:<6} -> {hours[c]:>9,.0f} GPU-h")

    if args.markdown:
        print()
        print("| model | A GPU-h | B GPU-h | C GPU-h | A days | B days | C days |")
        print("|---|---|---|---|---|---|---|")
        for model, h in per_model.items():
            print(
                f"| {model} | {h['full']:,.0f} | {h['public']:,.0f} | {h['mean']:,.0f} | "
                f"{h['full'] / args.gpus / 24:.1f} | {h['public'] / args.gpus / 24:.1f} | "
                f"{h['mean'] / args.gpus / 24:.1f} |"
            )
        print(
            f"| **total** | **{totals['full']:,.0f}** | **{totals['public']:,.0f}** | "
            f"**{totals['mean']:,.0f}** | **{totals['full'] / args.gpus / 24:.1f}** | "
            f"**{totals['public'] / args.gpus / 24:.1f}** | "
            f"**{totals['mean'] / args.gpus / 24:.1f}** |"
        )
        print()
        print("| | A paper-full | B public | C mean |")
        print("|---|---|---|---|")
        print(f"| SFT only GPU-h | {sft_totals['full']:,.0f} | {sft_totals['public']:,.0f} | {sft_totals['mean']:,.0f} |")
        print(
            f"| SFT only {args.gpus}-GPU days | {sft_totals['full'] / args.gpus / 24:.1f} | "
            f"{sft_totals['public'] / args.gpus / 24:.1f} | {sft_totals['mean'] / args.gpus / 24:.1f} |"
        )
        return

    print()
    for c in CALIBERS:
        print(
            f"{c:<7} {totals[c]:>10,.0f} GPU-h = {totals[c] / args.gpus:>8,.0f} "
            f"{args.gpus}-GPU hours = {totals[c] / args.gpus / 24:>6.1f} days"
        )
    print()
    for c in CALIBERS:
        print(
            f"SFT only ({c}): {sft_totals[c]:>9,.0f} GPU-h "
            f"({sft_totals[c] / args.gpus / 24:.1f} days)"
        )


if __name__ == "__main__":
    main()
