# Retrieval 模型评测对齐文档

三个检索模型（BGE-M3 / Qwen3-Embedding / Qwen3-Reranker）的**数据集、测试集、
评测指标**与原论文的对照。目的：让我们的复现结果能与论文数值直接比较。

原文依据：`Papers/M3-Embedding/main.tex`、`Papers/Qwen3-Embedding/contents/evaluation.tex`、
`Papers/CLIP/clip_paper.tex`（LaTeX 原文，最准确）。

## 索引

| 编号 | 文档 | 训练数据 | 评测 |
|---|---|---|---|
| 01 | [BGE-M3](01-bge-m3.md) | 预训练 1.2B（无监督）+ 微调 1.6M（含 T²-Ranking） | MIRACL / MKQA / MLDR / NarrativeQA |
| 02 | [Qwen3-Embedding](02-qwen3-embedding.md) | 弱监督 150M + 微调 7M 标注 + 12M 合成 | MMTEB 216 任务 |
| 03 | [Qwen3-Reranker](03-qwen3-reranker.md) | 无预训练，仅微调 7M + 12M | top-100 重排，6 个基准 |
| 04 | [CLIP](04-clip.md) | 预训练 WIT 4 亿对（未公开）<br>**我们的微调**：MSCOCO Captions 2017 train（21 shard / 19 GB，全参 3 epoch） | Flickr30k / MSCOCO 5k |
| 05 | [与当前实现的差异](05-gaps.md) | — | 逐项列出我们现在跑的 vs 论文要求 |
| 06 | [论文规模的 GPU-hours](06-gpu-hours.md) | — | 全规模 ~3.9-4.3 万；**只算可得数据 ~1,600** |
| **07** | [**A100 基准结果**](07-a100.md) | — | 基准平台：训练耗时、吞吐、完整检索指标（含 nDCG） |
| **08** | [**昆仑芯 P800 结果**](08-kunlunxin.md) | — | **精度 ✅ 99.4-100.3%；吞吐 Reranker ✅ / CLIP 需 1.51× / BGE 需 2.31×** |
| **09** | [**PPU ZW810E 结果**](09-ppu.md) | — | **精度 ✅ 99.9-100.1%；吞吐 CLIP ✅ / Reranker ✅ / BGE 差 0.18%** |
| **10** | [**微基准与折扣系数**](10-microbench-discount.md) | — | 六家芯片实测 bf16/带宽 → 折扣系数（含/不含 FP32 两口径） |
| **11** | [**给厂商的问题清单**](11-vendor-questions.md) | — | 量化后的剩余瓶颈 + 复现脚本 + 已排除方向 |

> 07-09 每份都含：训练耗时、吞吐（含最快/最慢/平均）、完整检索指标（含 nDCG）、
> docx §4 达标判定、**算子级剖析**（哪些算子拖慢、FlagGems 里有没有实现、
> 怎么接入）。
>
> **FlagGems 现状（实测）**：
> - 昆仑芯：厂商 Triton（`3.6.0+gitcd2d6c1b`，带 `language/extra/xpu`）与
>   flag_gems 都能装；但**实测 10 个算子里 9 个比原生更慢**（`sum` 慢 37×、
>   `layer_norm` 慢 26×），**换 kernel 这条路在昆仑芯上不成立**。
> - PPU：还没试。
>
> **优化目标必须按真实配置的整步 profiler 归因**（微基准和小 batch 外推都会给出
> 错误排序，详见 08）。昆仑芯 BGE 真实 step **1,419 → 2,923 ms（2.06×）**，
> 差距集中在：**GELU +988 ms（占差距 65.8%，慢 25.4×，走了 fp32
> `_highprecision` 路径）**、**dtype 转换/转置胶水 +309 ms（20.6%）**、
> gemm +153 ms（10.2%）。`flash_attention` 只有 0.86×（反而更快），
> `layer_norm` 0.80×（更快），**都不是瓶颈**。
>
> **step 里 94-96% 的时间是 device 计算，dataloader 只占 0.04%，
> tokenize 占 3-6%。瓶颈在算子，不在数据管线。**
>
> **全部结果均未引入 FlagGems**（`use_flaggems=false`），属未优化基线。

## 一句话结论

**训练侧和评测侧都无法与论文对齐。**

### 训练侧：三个模型都不可精确复现

| 模型 | 缺什么 | 占训练数据比例 |
|---|---|---|
| BGE-M3 | 预训练 MTP（内部数据）+ MultiLongDoc（GPT-3.5 合成） | ~24% 预训练量 |
| Qwen3-Embedding | 全部合成数据（150M 预训练 + 12M 微调）+ 清单有 `.etc` | **63% 的微调数据** |
| Qwen3-Reranker | 同上 | 同上 |
| CLIP | WIT 4 亿对（自建爬取，从未发布） | 全部 |

论文都只承诺开源**模型权重**，没有承诺开源训练数据。我们实际做的是
**"从官方预训练 checkpoint 出发，在公开数据上做增量微调"**，
报告时必须按这个口径写，不能说"复现了论文的训练"。

### 评测侧：口径不一致，但指标已补齐

| | 论文 | 我们 |
|---|---|---|
| 评测集 | MIRACL / MMTEB / MTEB-R 等 | **T²-Ranking test split**（论文的训练分布内） |
| 候选池 | 全语料，或 reranker 用 embedding 的 top-100 | 每 query **16 候选** |
| 主指标 | nDCG@10（文本）/ R@1/5/10（CLIP） | **nDCG@{1,5,10,20} + recall + hit + MRR** ✅ 已补 |

指标已能覆盖论文主指标（nDCG 于 2026-10-29 加入）。**真正的缺口是候选池**
（16 vs 全语料），那个要改评测协议，不只是加指标。

详见 [05-gaps.md](05-gaps.md)。

## 吞吐（样本/s，含波动区间）

吞吐定义按 docx：训练集总样本数 ÷ 训练总时间。波动区间从 wandb 的
`_timestamp` + `optimizer_step` 逐区间推导（去掉首尾 5% 的启动/收尾开销）。

| 模型 | 平台 | 最慢 | 平均（区间） | 端到端 | 最快 | 数据源 |
|---|---|---|---|---|---|---|
| BGE-M3 | 昆仑芯 | **43.1** | **56.4** | 54.0 | **60.4** | wandb `fw8k8bib` |
| | PPU | — | — | 67.0 | — | ❌ wandb 未记录 |
| | A100 | — | — | 118.0 | — | ❌ wandb 未记录 |
| CLIP | 昆仑芯 | **72.2** | **85.3** | 83.4 | **87.7** | wandb `epw35qs6` |
| | PPU | — | — | 69.1 | — | ❌ wandb 未记录 |
| | A100 | — | — | 119.0 | — | ❌ wandb 未记录 |
| Reranker（2 卡） | 昆仑芯 | **5.0** | **6.4** | — | **6.5** | wandb `ivfpp1e8` |
| | PPU | **3.7** | **3.9** | — | **4.2** | wandb `eoxqu2xl` |
| | A100 | — | — | ~3.0 | — | ❌ 训练提前停止 |

> A100 与 PPU 的 bge/clip 那几次 run 都早于 `WANDB_API_KEY` 生效，
> wandb 全程离线，日志也没有每步耗时，**只能给端到端值**。
> 补齐需要重跑并确认 wandb 正常，或在日志里加逐段时间戳。

## 达标判定（docx §4）

> ⚠️ **以下所有数字均未引入 FlagGems**（`use_flaggems=false`，走厂商默认算子）。
> 这是**未优化的基线**：吞吐未达标的项，正是 FlagGems 要补的；
> 精度不受影响（算子替换不改数值语义，但仍需回归验证）。

| 要求 | A100 | 昆仑芯 | PPU |
|---|---|---|---|
| **精度** ≥ A100 的 95%（BGE / CLIP） | 基准 | ✅ **99.4-100.3%** | ✅ **99.9-100.1%** |
| **精度**（Reranker） | 基准 | ⚠️ 无可用 checkpoint | ⚠️ 训练中 |
| **吞吐**（BGE） | 118.0 | ❌ **43.3%**（54.0） | ⚠️ **99.8%**（67.0） |
| **吞吐**（CLIP） | 119.0 | ❌ **66.3%**（83.4） | ✅ **102.1%**（69.1） |
| **吞吐**（Reranker，2 卡） | **5.51** | ✅ **109.9%**（6.4） | ✅ **124.4%**（3.9） |

> 单位：bge/clip = 样本/s（单卡），reranker = 对/s（2 卡合计）。
> 百分比 = 实测 ÷ (A100 × 折扣系数 × 0.8)，**≥100% 即达标**。
> 系数取实测微基准（不含 FP32）：昆仑芯 **1.321**、PPU **0.711**，
> 详见 [10-microbench-discount.md](10-microbench-discount.md)。

**折扣系数该用 bf16 还是 fp32？→ bf16。** 实测三个模型的 fp32 GEMM 占比分别是
**0.18% / 0% / 0.001%**（详见 `07-a100.md`），负载本质是 bf16-GEMM；
`master_dtype=fp32` 只影响常驻显存与 optimizer，不影响前向/反向的 GEMM。
这正好回应了 docx §4.2 对 fp32 FLOPS 口径的质疑：**该口径不适用于本工作负载。**

**结论：**

1. **精度**：bge/clip 全面达标（99.4-100.3%），余量很大。reranker 两平台都缺数据。
2. **吞吐（按实测折扣系数）**：**5 项中 3 项达标**
   - ✅ 昆仑芯 reranker（109.9%）、PPU CLIP（102.1%）、PPU reranker（124.4%）
   - ⚠️ PPU BGE 差 **0.18%**（差 0.12 样本/s）
   - ❌ 昆仑芯 CLIP 需提速 1.51×、BGE 需提速 2.31×
3. 之前按"系数=1"的保守判定（三项全不达标）会**低估国产芯片**——
   昆仑芯 bf16 GEMM 实测是 A100 的 127%、带宽 138%。

详见 [07](07-a100.md) / [08](08-kunlunxin.md) / [09](09-ppu.md)。

## 一个容易混淆的点：kernel 慢 ≠ 吞吐慢

profiler 的「每 step device 时间」和 docx 的「吞吐」是两个口径。下表的
三列均来自**按阶段实测**（`examples/retrieval/profile_step_segments.py`，
BGE-M3 真实配置 batch 10 × accum 4 = 200 texts/step）：

| | BGE-M3 A100 | 昆仑芯 | PPU |
|---|---|---|---|
| 整 step 墙钟 | **1,605 ms** | **3,262 ms**（2.03×） | **2,549 ms**（1.59×） |
| 其中 device 计算（fwd+bwd） | **1,504 ms（93.7%）** | **3,132 ms（96.0%）** | **2,556 ms（100.3%）** |
| 其中 tokenize | 107 ms（6.7%） | 105 ms（3.2%） | **67 ms（2.6%）** |
| 其中 **dataloader** | **0.67 ms（0.04%）** | **1.11 ms（0.03%）** | **0.78 ms（0.03%）** |
| 其中 clip + optimizer | 22 ms（1.4%） | 62 ms（1.9%） | 22 ms（0.9%） |
| kernel（profiler 单层） | 1,419 ms | 2,923 ms | 2,531 ms |
| 吞吐（整轮训练） | 117.9 texts/s | 54.0（45.8%） | 66.6（56.6%） |

**之前这里写的是"两个 step 都被非 kernel 开销主导"，那是错的。**
旧数字（A100 kernel 252 ms = 15%、昆仑芯 1,751 ms = 47%）是拿
「batch 4 × seq 512 的合成 batch、一次 forward+backward」的 profile
乘以调用次数外推出来的，而且把 aten 层和 kernel 层重复相加了。
按真实配置实测，**三个平台的 step 都有 94-100% 在 device 计算里**，
数据管线只占 0.03-0.04%。

由此得到一个更明确的结论：**这个 workload 的瓶颈就是算子，不是数据管线。**
优化算子是对的方向；把 num_workers 调大、给 collate 上缓存几乎不会有收益
（可优化空间上限就是那 7% 的 tokenize）。

**三家的短板几乎不重叠**（按占差距的比例）：

| 平台 | 头号 | 第二 | 第三 | 反而更快的 |
|---|---|---|---|---|
| 昆仑芯 | GELU **65.8%** | cast/transpose 20.6% | GEMM 10.2% | attention 0.86×、norm 0.80× |
| PPU | GEMM **44.3%** | attention 20.4% | cast/copy 14.7% | GELU 0.9×、norm 0.6× |

所以**不能用一套优化手段同时打两家**：昆仑芯的 GELU 是 bug 级的 25×，
PPU 的 GEMM 是 1.7× 的效率差，且调用次数与 A100 完全相同。

