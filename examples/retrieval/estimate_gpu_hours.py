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

# texts/s or pairs/s on ONE A100-SXM4-80GB. All measured end to end by us.
THROUGHPUT = {
    # 07-a100.md: 23,010 steps / 39,017.5 s, 200 texts per step.
    "bge_512": 118.0,
    # Extrapolated from bge_512 by sequence length (512 -> 8192). NOT measured:
    # attention is O(L^2), so the true value can be 2x lower. Biggest uncertainty
    # in the whole estimate; unblock by running ~100 RetroMAE steps at 8192.
    "bge_8192": 7.4,
    # 07-a100.md: 5.51 pairs/s across 2 cards, so 2.755 per card.
    "reranker_640": 5.51 / 2,
    # 07-a100.md: 318,720 samples / 2,679.1 s.
    "clip_77": 119.0,
}


@dataclass(frozen=True)
class Stage:
    model: str
    phase: str  # "pretrain" or "sft"
    stage: str
    rate: str
    unit: str
    # Size under each caliber, already converted to the unit the rate is in
    # (a text pair costs two encoder passes, so pairs are doubled for bi-encoders).
    full: float
    public: float
    mean: float
    note: str = ""
    # Epoch count the paper itself trains for. None means the paper never states
    # one (BGE-M3 says only "epoch" nowhere; the Qwen3 report gives no steps at
    # all), so an assumption is supplied in PAPER_ASSUMPTION below.
    paper_epochs: float | None = None
    # Some stages are pinned by an absolute sample count rather than by epochs:
    # BGE-M3's appendix gives 20,000 / 25,000 steps, which fixes the total work
    # no matter how much data you actually hold. Expressed in `unit` terms.
    paper_absolute: float | None = None
    # True when paper_epochs is our assumption, not the paper's statement.
    assumed: bool = False


# Divisors are spelled out in the note so the mean caliber can be re-checked.
# Covers exactly the three models in scope: BGE-M3, CLIP, Qwen3-Reranker.
STAGES = [
    Stage(
        "BGE-M3", "pretrain", "pretrain-A RetroMAE@8192", "bge_8192", "texts",
        full=184e6, public=184e6, mean=184e6 / 3,
        note="Pile/Wudao/mC4, all public -> A == B; mean divides by 3 sources",
        # 20,000 steps x (32 per-GPU x GA 16 x 32 GPUs) = 327.68M samples
        # = 1.78 passes of the 184M set. Absolute, so caliber-independent.
        paper_epochs=327.68e6 / 184e6,
        paper_absolute=327.68e6,
    ),
    Stage(
        "BGE-M3", "pretrain", "pretrain-B contrastive@512", "bge_512", "texts",
        full=1.2e9 * 2, public=928.3e6 * 2, mean=928.3e6 / 4 * 2,
        note="77% public (MTP 291.1M missing); mean divides by 4 public tables",
        # 25,000 steps x 67,200 (paper's total batch for the 0-500 bucket)
        # = 1.68B pairs = 1.40 passes of 1.2B. The "32 per GPU" reading gives
        # 0.064 passes instead -- a 22x spread the paper does not resolve.
        paper_epochs=1.68e9 / 1.2e9,
        paper_absolute=1.68e9 * 2,
    ),
    Stage(
        "BGE-M3", "sft", "SFT@512", "bge_512", "texts",
        full=1.648e6 * 9, public=1.5755e6 * 9, mean=1.5755e6 / 16 * 9,
        note="9 texts/row (1 query + 1 pos + 7 neg); 17 sets full, 16 public",
        # Paper gives only "~6,000 steps to perform warm-up"; the unified
        # training length is never stated, so 3 epochs is our assumption.
        paper_epochs=3,
        assumed=True,
    ),
    Stage(
        "Qwen3-Rerank-8B", "pretrain", "none (no pretrain stage)", "reranker_640", "pairs",
        full=0, public=0, mean=0,
        note="paper starts from the Qwen3 base; nothing to walk",
    ),
    Stage(
        "Qwen3-Rerank-8B", "sft", "SFT@640", "reranker_640", "pairs",
        full=19e6, public=2.8e6, mean=2.8e6 / 12,
        note="7M labeled + 12M synthetic; public = 40% of the 7M over 12 sets",
        # No epoch, step count, GPU count or batch size is published. 3 is ours.
        paper_epochs=3,
        assumed=True,
    ),
    Stage(
        "CLIP ViT-L/14", "pretrain", "pretrain@77", "clip_77", "pairs",
        full=400e6, public=15e6, mean=15e6,
        note="WIT unreleased; YFCC filtered to 15M is the paper's own substitute",
        # Stated outright: "We train all models for 32 epochs". Corroborated by
        # "12.8 billion images seen over 32 training epochs" (400M x 32).
        paper_epochs=32,
    ),
    Stage(
        "CLIP ViT-L/14", "sft", "SFT@77 (not in paper)", "clip_77", "pairs",
        full=0, public=106240, mean=106240,
        note="no such stage in the paper; our MSCOCO finetune",
        paper_epochs=3,
        assumed=True,
    ),
]

CALIBERS = ("full", "public", "mean")
PHASES = ("pretrain", "sft")


def gpu_hours(units: float, rate_per_s: float) -> float:
    return units / (rate_per_s * 3600.0)


def fmt(x: float) -> str:
    if x >= 1:
        return f"{x:,.0f}"
    return f"{x:.1f}" if x > 0 else "0"


def paper_epoch_tables(gpus: int) -> None:
    """Cost each stage at the epoch count the paper states, not at one pass.

    Two kinds of stage, and they behave differently across calibers:

    * CLIP / reranker publish epochs, so cost = epochs x caliber size. Halving
      the available data halves the bill.
    * BGE-M3 publishes steps, which pin absolute work (``paper_absolute``).
      The bill is identical under every caliber; only the number of passes over
      the available data changes. Multiplying by the caliber-1-epoch cost would
      be wrong here, which is why the absolute count is used instead.
    """
    print("### 论文声明的 epoch 数（原文核查）\n")
    print("| model | stage | paper epoch | 来源 |")
    print("|---|---|---|---|")
    for st in STAGES:
        if st.paper_epochs is None:
            continue
        ep = f"{st.paper_epochs:.2f}"
        if st.assumed:
            src = "**未给，本表假设**"
        elif st.paper_absolute is not None:
            src = "由论文步数反推"
        else:
            src = "论文明确声明"
        print(f"| {st.model} | {st.stage} | {ep} | {src} |")

    print("\n### 按论文 epoch 的成本（单机 %d 卡）\n" % gpus)
    print("| model | stage | A 论文全量 | B 公开可得 | C 平均单集 |")
    print("|---|---|---|---|---|")

    totals = {c: 0.0 for c in CALIBERS}
    for st in STAGES:
        if st.paper_epochs is None:
            continue
        cost = {}
        for c in CALIBERS:
            if st.paper_absolute is not None:
                units = st.paper_absolute
            else:
                units = getattr(st, c) * st.paper_epochs
            cost[c] = gpu_hours(units, THROUGHPUT[st.rate])
            totals[c] += cost[c]
        cells = " | ".join(f"{fmt(cost[c])} / {cost[c] / gpus / 24:.1f} d" for c in CALIBERS)
        print(f"| {st.model} | {st.stage} | {cells} |")

    cells = " | ".join(
        f"**{totals[c]:,.0f} / {totals[c] / gpus / 24:.1f} d**" for c in CALIBERS
    )
    print(f"| **合计** | | {cells} |")
    print()
    for c in ("full", "public", "mean"):
        h = totals[c]
        print(f"{c:<7} {h:>9,.0f} GPU-h = {h / gpus:>7,.0f} "
              f"{gpus}-GPU h = {h / gpus / 24:>6.1f} days")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", action="store_true", help="emit markdown tables")
    ap.add_argument("--gpus", type=int, default=8, help="GPUs per server")
    ap.add_argument(
        "--paper-epochs",
        action="store_true",
        help="cost each stage at the epoch count the paper declares, not 1",
    )
    args = ap.parse_args()

    if args.paper_epochs:
        paper_epoch_tables(args.gpus)
        return

    totals = {c: 0.0 for c in CALIBERS}
    phase_totals = {p: {c: 0.0 for c in CALIBERS} for p in PHASES}
    # (model, phase) -> caliber -> GPU-h. Pretrain and SFT stay separate rows,
    # because they are separately schedulable (and SFT is often skipped entirely).
    cell: dict[tuple[str, str], dict[str, float]] = {}
    model_totals: dict[str, dict[str, float]] = {}

    if args.markdown:
        print("### Per stage（单卡 A100 实测吞吐 × 数据量 ÷ 3600）\n")
        print("| model | phase | stage | unit | A paper-full | B public | C mean |")
        print("|---|---|---|---|---|---|---|")

    for st in STAGES:
        hours = {c: gpu_hours(getattr(st, c), THROUGHPUT[st.rate]) for c in CALIBERS}
        c = cell.setdefault((st.model, st.phase), {x: 0.0 for x in CALIBERS})
        mt = model_totals.setdefault(st.model, {x: 0.0 for x in CALIBERS})
        for x in CALIBERS:
            totals[x] += hours[x]
            c[x] += hours[x]
            mt[x] += hours[x]
            phase_totals[st.phase][x] += hours[x]

        if args.markdown:
            print(
                f"| {st.model} | {st.phase} | {st.stage} | {st.unit} | "
                + " | ".join(fmt(hours[x]) for x in CALIBERS)
                + " |"
            )
        else:
            print(f"{st.model} / {st.phase} / {st.stage}  [{st.note}]")
            for x in CALIBERS:
                print(f"    {x:<7} {getattr(st, x):>16,.0f} {st.unit:<6} -> {hours[x]:>9,.0f} GPU-h")

    def row(label: str, h: dict[str, float], bold: bool = False) -> str:
        vals = []
        for x in CALIBERS:
            days = h[x] / args.gpus / 24
            v = f"{h[x]:,.0f} GPU-h / {days:.1f} d"
            vals.append(f"**{v}**" if bold else v)
        return f"| {label} | " + " | ".join(vals) + " |"

    if args.markdown:
        print("\n### 两阶段分列（单机 8 卡 A100）\n")
        print("| 模型 | 阶段 | A 论文全量 | B 公开可得 | C 平均单集 |")
        print("|---|---|---|---|---|")
        for model in model_totals:
            for phase in PHASES:
                key = (model, phase)
                if key not in cell:
                    continue
                label = "预训练" if phase == "pretrain" else "微调 SFT"
                print(row(f"{model} · {label}", cell[key]))
        print("\n### 阶段合计\n")
        print("| 阶段 | A 论文全量 | B 公开可得 | C 平均单集 |")
        print("|---|---|---|---|")
        for phase in PHASES:
            label = "预训练" if phase == "pretrain" else "微调 SFT"
            print(row(label, phase_totals[phase]))
        print(row("**总计**", totals, bold=True))
        return

    print()
    for model in model_totals:
        for phase in PHASES:
            key = (model, phase)
            if key not in cell:
                continue
            h = cell[key]
            print(
                f"{model:<16}{'预训练' if phase == 'pretrain' else 'SFT':<8}"
                f"full {h['full']:>8,.0f}  public {h['public']:>8,.0f}  mean {h['mean']:>8,.0f} GPU-h"
            )
    print()
    for phase in PHASES:
        label = "pretrain" if phase == "pretrain" else "sft"
        for x in CALIBERS:
            h = phase_totals[phase][x]
            print(f"{label:<9}{x:<7} {h:>9,.0f} GPU-h = {h / args.gpus:>7,.0f} "
                  f"8-GPU h = {h / args.gpus / 24:>6.1f} days")
    print()
    for x in CALIBERS:
        h = totals[x]
        print(f"{'TOTAL':<16}{x:<7} {h:>9,.0f} GPU-h = {h / args.gpus:>7,.0f} "
              f"8-GPU h = {h / args.gpus / 24:>6.1f} days")


if __name__ == "__main__":
    main()
