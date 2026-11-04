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
    # 07-a100.md: 7.62 pairs/s across 2 cards (16.8 s/step, 128 pairs/step),
    # so 3.81 per card.  The old 5.51 was wrong twice over: it used 7,670
    # steps/epoch where the log shows 4,790, and counted 64 *rows* per step as
    # 64 *pairs* although each row yields one positive plus one negative.
    "reranker_640": 7.62 / 2,
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
    # ``_lo`` / ``_hi`` carry the range the paper leaves open.
    paper_absolute: float | None = None
    paper_absolute_lo: float | None = None
    paper_absolute_hi: float | None = None
    # True when paper_epochs is our assumption, not the paper's statement.
    assumed: bool = False


# Divisors are spelled out in the note so the mean caliber can be re-checked.
# Covers exactly the three models in scope: BGE-M3, CLIP, Qwen3-Reranker.
STAGES = [
    Stage(
        "BGE-M3", "pretrain", "pretrain-A RetroMAE@8192", "bge_8192", "texts",
        full=184e6, public=184e6, mean=184e6 / 3,
        note="Pile/Wudao/mC4, all public -> A == B; mean divides by 3 sources",
        # 20,000 steps x (32 per-GPU x GA 16 x 32 GPUs) = 327.68M samples, i.e.
        # 1.78 passes of the 184M set. "batch size 32" must be the per-device
        # micro-batch: 32 GPUs x GA 16 puts the global batch at >= 512, so a
        # global reading of 32 is not even expressible. See docs section 11.2.
        paper_epochs=327.68e6 / 184e6,
        paper_absolute=327.68e6,
    ),
    Stage(
        "BGE-M3", "pretrain", "pretrain-B contrastive@512", "bge_512", "texts",
        full=1.2e9 * 2, public=928.3e6 * 2, mean=928.3e6 / 8 * 2,
        note="77% public (MTP 291.1M missing); mean divides by the 8 public "
             "datasets in DATASETS, not the paper's 4 grouping rows",
        # 25,000 steps. Table 8's column header reads "Detailed total batch
        # size", so these are global batches, ranging 9,984 (7k-8k tokens) to
        # 67,200 (0-500 tokens). 25,000 steps therefore = 249.6M to 1.68B pairs.
        # We use the nine-bucket mean (29,216/step) as the central estimate.
        paper_epochs=0.609,
        paper_absolute=29_216 * 25_000 * 2,
        paper_absolute_hi=67_200 * 25_000 * 2,
        paper_absolute_lo=9_984 * 25_000 * 2,
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

# Per-dataset breakdown for calibers A and B. Sizes come from the papers:
# M3-Embedding Table 1 (appendix) for BGE-M3, section 5 for CLIP, and the Qwen3
# report for the reranker. ``full`` / ``public`` are separate sizes because they
# genuinely differ (WIT has no public equivalent; the reranker's named sets are
# only partly public). Caliber C is deliberately absent: it is the public total
# divided by the dataset count, so listing datasets would restate that division.
DATASETS = [
    # model, phase, dataset, unit, size_full, size_public, note
    ("BGE-M3", "pretrain", "Pile + Wudao + mC4（RetroMAE）", "texts",
     184e6, 184e6, "论文只给合计 184M / 105 语言，无逐源拆分"),
    ("BGE-M3", "pretrain", "MTP", "pairs", 291.1e6, 0, "BGE 内部数据，未公开"),
    ("BGE-M3", "pretrain", "S2ORC", "pairs", 41.8e6, 41.8e6, "源码注释里的逐项值"),
    ("BGE-M3", "pretrain", "Wikipedia", "pairs", 6.5e6, 6.5e6, "48.3M 减 S2ORC 41.8M"),
    ("BGE-M3", "pretrain", "xP3", "pairs", 62.6e6, 62.6e6, "源码注释里的逐项值"),
    ("BGE-M3", "pretrain", "mC4", "pairs", 16.4e6, 16.4e6, "源码注释里的逐项值"),
    ("BGE-M3", "pretrain", "CC-News", "pairs", 409.4e6, 409.4e6, "源码注释里的逐项值"),
    ("BGE-M3", "pretrain", "NLLB", "pairs", 197.8e6, 197.8e6, "源码注释里的逐项值"),
    ("BGE-M3", "pretrain", "CCMatrix", "pairs", 193.5e6, 193.5e6, "源码注释里的逐项值"),
    ("BGE-M3", "pretrain", "CodeSearchNet", "pairs", 0.3441e6, 0.3441e6, "文本-代码"),
    ("BGE-M3", "sft", "EN 8 集", "rows", 1.1e6, 1.1e6,
     "HotpotQA/TriviaQA/NQ/MS MARCO/COLIEE/PubMedQA/SQuAD/SimCSE-NLI"),
    ("BGE-M3", "sft", "ZH 7 集", "rows", 386.6e3, 386.6e3,
     "DuReader/mMARCO-ZH/T²-Ranking/LawGPT/CMedQAv2/NLI-zh/LeCaRDv2"),
    ("BGE-M3", "sft", "多语 2 集", "rows", 88.9e3, 88.9e3, "Mr.TyDi / MIRACL"),
    ("BGE-M3", "sft", "MultiLongDoc（合成）", "rows", 41.4e3, 0, "仅公布 prompt"),
    ("CLIP ViT-L/14", "pretrain", "WIT", "pairs", 400e6, 0, "自建爬取，未公开"),
    ("CLIP ViT-L/14", "pretrain", "YFCC100M 过滤后", "pairs", 0, 15e6,
     "替代方案：原始 100M，过滤后约 15M"),
    ("CLIP ViT-L/14", "sft", "MSCOCO Captions 2017 train", "pairs", 0, 106240,
     "论文无此阶段，我们自加"),
    ("Qwen3-Rerank-8B", "sft", "标注 12 个点名集 + .etc", "pairs",
     7e6, 2.8e6, "论文只给 7M 总数；公开部分按 40% 假设，见 9.2"),
    ("Qwen3-Rerank-8B", "sft", "高质量合成（cos>0.7 筛出）", "pairs",
     12e6, 0, "未发布"),
]

# How many encoder passes one unit costs. BGE-M3 is a bi-encoder, so a pair is
# two encodes and an SFT row is 1 query + 1 pos + 7 neg = 9 encodes. CLIP scores
# pairs jointly and the reranker is a cross-encoder, so one pair = one pass --
# the rates for those two are already per pair, not per text.
UNIT_MULTIPLIER = {
    "BGE-M3": {"texts": 1, "pairs": 2, "rows": 9},
    "CLIP ViT-L/14": {"pairs": 1},
    "Qwen3-Rerank-8B": {"pairs": 1},
}


def gpu_hours(units: float, rate_per_s: float) -> float:
    return units / (rate_per_s * 3600.0)


def fmt(x: float) -> str:
    if x >= 1:
        return f"{x:,.0f}"
    return f"{x:.1f}" if x > 0 else "0"


def stage_rate(model: str, phase: str, dataset: str) -> str:
    """Sequence length decides the rate: RetroMAE runs at 8192, the rest at 512."""
    if "RetroMAE" in dataset:
        return "bge_8192"
    if "Rerank" in model:
        return "reranker_640"
    if "CLIP" in model:
        return "clip_77"
    return "bge_512"


# Caliber D: for each stage, the smallest single dataset that can still run it.
# Sizes are one dataset, not a group -- where the paper only publishes a group
# total, the group is divided by its member count and the note says so.
MIN_DATASET = [
    # model, phase, dataset, unit, size, multiplier-from-group, note
    ("BGE-M3", "pretrain", "Pile / Wudao / mC4 之一（RetroMAE）", "texts",
     184e6 / 3, "论文只给 184M 合计，按 3 源均分；这是全流程的时间下限"),
    ("BGE-M3", "pretrain", "CodeSearchNet（对比阶段）", "pairs",
     0.3441e6, "全表最小"),
    ("BGE-M3", "sft", "Mr.TyDi 或 MIRACL 之一", "rows",
     88.9e3 / 2, "88.9K 是两集合计，除以 2"),
    ("CLIP ViT-L/14", "pretrain", "YFCC100M 过滤后", "pairs",
     15e6, "唯一公开替代，无更小的选项"),
    ("CLIP ViT-L/14", "sft", "MSCOCO Captions 2017 train", "pairs",
     106240, "唯一选项"),
    ("Qwen3-Rerank-8B", "sft", "12 点名集里最小的一个", "pairs",
     2.8e6 / 12, "2.8M 是 12 集合计，除以 12"),
]

# How many passes over each dataset the paper's own schedule amounts to.
# BGE's two pretraining stages are pinned by step counts, so their pass count is
# derived rather than stated; the SFT and reranker schedules are not published at
# all, so 3 passes is our assumption (marked with * in the output).
PAPER_PASSES = {
    ("BGE-M3", "pretrain"): 327.68e6 / 184e6,   # 20,000 steps, 1.78 passes
    ("BGE-M3", "sft"): 3.0,                      # assumed
    ("CLIP ViT-L/14", "pretrain"): 32.0,         # stated by the paper
    ("CLIP ViT-L/14", "sft"): 3.0,               # assumed
    ("Qwen3-Rerank-8B", "sft"): 3.0,             # assumed
}
ASSUMED_PASSES = {("BGE-M3", "sft"), ("CLIP ViT-L/14", "sft"), ("Qwen3-Rerank-8B", "sft")}


# 25,000 steps x 29,216 (mean of the paper's nine length buckets) = 730.4M pairs
# drawn from the 1.2B pool. The contrastive stage therefore has an ABSOLUTE
# sample budget, so each dataset's share is that budget divided by the caliber
# total -- not a fixed per-dataset pass count.
CONTRASTIVE_PAIRS_DRAWN = 29_216 * 25_000


def paper_passes(model: str, phase: str, dataset: str, caliber_pool: float) -> float:
    """Passes over one dataset implied by the paper's schedule.

    Every stage except the contrastive one is a per-dataset repeat count. The
    contrastive stage draws a fixed 730.4M pairs from whichever pool you hold,
    so a smaller caliber means more passes over each of its datasets for the
    same total work -- which is why its cost is caliber-independent.
    """
    if model == "BGE-M3" and phase == "pretrain" and "RetroMAE" not in dataset:
        return CONTRASTIVE_PAIRS_DRAWN / caliber_pool
    return PAPER_PASSES[(model, phase)]


def minimum_tables(gpus: int) -> None:
    """Caliber D: one smallest dataset per stage -- the shortest full pipeline.

    Answers 'how little can I get away with and still exercise every stage'. The
    contrastive stage has no paper-volume figure: its schedule draws 730.4M
    pairs, and the smallest single dataset holds 0.34M, so honouring it would
    mean over 2,000 passes over the same tiny file.
    """
    print("### 口径 D：每阶段只跑最小的那一个数据集\n")
    print("| 模型 | 阶段 | 最小数据集 | 数据量 | 单遍 GPU-h | 最小单位耗时 | 备注 |")
    print("|---|---|---|---|---|---|---|")
    total = 0.0
    for model, phase, name, unit, size, note in MIN_DATASET:
        rate = stage_rate(model, phase, name)
        hours = gpu_hours(size * UNIT_MULTIPLIER[model][unit], THROUGHPUT[rate])
        total += hours
        unit_cn = {"texts": "条文本", "pairs": "对", "rows": "行"}[unit]
        shown = f"{size / 1e6:,.2f}M {unit_cn}"
        wall = f"{hours / gpus * 60:.1f} 分钟" if hours < 1 else f"{hours / gpus:.1f} 机器时"
        print(f"| {model} | {phase} | {name} | {shown} | {hours:,.2f} | {wall} | {note} |")
    print(f"\n**合计 {total:,.0f} GPU-h = {total / gpus:.1f} 机器时 = "
          f"{total / gpus / 24:.1f} 天**（单机 {gpus} 卡）\n")
    print("按论文 epoch 数（仅列能成立的阶段）：\n")
    print("| 模型 | 阶段 | 遍数 | GPU-h | 备注 |")
    print("|---|---|---|---|---|")
    ptotal = 0.0
    for model, phase, name, unit, size, note in MIN_DATASET:
        if model == "BGE-M3" and phase == "pretrain" and "RetroMAE" not in name:
            print(f"| {model} | {phase} | — | — | 论文要采 730.4M 对，最小集只有 "
                  f"{size / 1e6:.2f}M，需 2,123 遍，不成立 |")
            continue
        rate = stage_rate(model, phase, name)
        passes = paper_passes(model, phase, name, size)
        hours = gpu_hours(size * UNIT_MULTIPLIER[model][unit] * passes, THROUGHPUT[rate])
        ptotal += hours
        print(f"| {model} | {phase} | {passes:.2f}× | {hours:,.1f} | {note} |")
    print(f"\n**合计 {ptotal:,.0f} GPU-h = {ptotal / gpus / 24:.1f} 天**\n")


def dataset_tables(gpus: int) -> None:
    """Per-dataset sizes and cost, for calibers A and B separately.

    Caliber C is omitted on purpose: it is the public total divided by the
    number of datasets, so a per-dataset list would just restate that division.
    """
    for caliber, field, title in (
        ("full", 3, "A 论文全量"),
        ("public", 4, "B 公开可得"),
    ):
        print(f"### 口径 {title}：逐数据集\n")
        print("| 模型 | 阶段 | 数据集 | 数据量 | 单遍 GPU-h | "
              "论文遍数 | 论文量 GPU-h | 备注 |")
        print("|---|---|---|---|---|---|---|---|")
        # Pool the contrastive stage samples from, for this caliber.
        pool_contrastive = (
            sum(r[field - 3] for mo, ph, n, u, *r in DATASETS
                if mo == "BGE-M3" and ph == "pretrain" and "RetroMAE" not in n)
        )
        subtotal = 0.0
        paper_subtotal = 0.0
        for model, phase, name, unit, *rest in DATASETS:
            size = rest[field - 3]
            note = rest[-1]
            if size <= 0:
                continue
            rate = stage_rate(model, phase, name)
            multiplier = UNIT_MULTIPLIER[model][unit]
            hours = gpu_hours(size * multiplier, THROUGHPUT[rate])
            passes = paper_passes(model, phase, name, pool_contrastive)
            paper_hours = hours * passes
            subtotal += hours
            paper_subtotal += paper_hours
            unit_cn = {"texts": "条文本", "pairs": "对", "rows": "行"}[unit]
            shown = f"{size / 1e6:,.1f}M {unit_cn}"
            if unit == "rows":
                shown += " × 9 texts"
            # * marks a pass count we assumed, not one the paper states
            mark = " †" if (model, phase) in ASSUMED_PASSES else ""
            print(f"| {model} | {phase} | {name} | {shown} | {hours:,.1f} | "
                  f"{passes:.2f}×{mark} | {paper_hours:,.1f} | {note} |")
        print(f"\n**小计：单遍 {subtotal:,.0f} GPU-h；论文量 {paper_subtotal:,.0f} GPU-h "
              f"= {paper_subtotal / gpus / 24:.1f} 天**  "
              f"（† = 遍数是假设值，论文未给）\n")


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
        cells = []
        for c in CALIBERS:
            if st.paper_absolute is not None and st.paper_absolute_hi:
                lo = gpu_hours(st.paper_absolute_lo or st.paper_absolute, THROUGHPUT[st.rate])
                hi = gpu_hours(st.paper_absolute_hi, THROUGHPUT[st.rate])
                cells.append(f"{lo:,.0f} ~ {hi:,.0f} / {lo / gpus / 24:.1f}-{hi / gpus / 24:.1f} d")
            else:
                cells.append(f"{fmt(cost[c])} / {cost[c] / gpus / 24:.1f} d")
        print(f"| {st.model} | {st.stage} | " + " | ".join(cells) + " |")

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
        "--minimum",
        action="store_true",
        help="caliber D: one smallest dataset per stage",
    )
    ap.add_argument(
        "--datasets",
        action="store_true",
        help="per-dataset sizes for calibers A and B",
    )
    ap.add_argument(
        "--paper-epochs",
        action="store_true",
        help="cost each stage at the epoch count the paper declares, not 1",
    )
    args = ap.parse_args()

    if args.minimum:
        minimum_tables(args.gpus)
        return
    if args.datasets:
        dataset_tables(args.gpus)
        return
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
