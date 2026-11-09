# 02 · Qwen3-Embedding

原文：`Papers/Qwen3-Embedding/contents/evaluation.tex`，2025，§3–4、表 6–9。
模型权重：`Qwen/Qwen3-Embedding-0.6B`（我们用的是 0.6B）。

## 训练数据（论文表 5，两阶段）

### 阶段一：弱监督预训练

| 来源 | 规模 | 可得性 | 说明 |
|---|---|---|---|
| Qwen3-32B 合成 | **~150M 对** | ❌ **未发布** | 只给了 prompt 模板（附录），数据本身拿不到 |

合成方式：从 Qwen3 多语预训练语料取文档 → 检索模型从 Persona Hub 选 top-5
"角色候选" → 32B 模型生成该文档可能的用户 query，可控 question_type
（keywords / acquire_knowledge / summary / yes_or_no / background）、
difficulty、length、language。

### 阶段二：监督微调

论文表 5 的原文（**注意结尾的 `.etc`**）：

```latex
MS MARCO, NQ, HotpotQA, NLI, Dureader, T²-Ranking,
SimCLUE, MIRACL, MLDR, Mr.TyDi, Multi-CPR, CodeSearchNet .etc
+ High-quality Synthetic Data
```

| 来源 | 规模 | 可得性 |
|---|---|---|
| 标注数据（上列 12 个 + `.etc` 未公开部分） | **~7M** | ⚠️ 12 个公开，其余未知 |
| 高质量合成（从 150M 里按 cos>0.7 筛出） | **~12M** | ❌ **未发布** |

微调后做了 **model merging**（slerp 球面线性插值，合并多个 checkpoint）。

> ⚠️ **两个都拿不到**：合成数据占微调数据的 **63%**（12M / 19M），
> 且清单本身有 `.etc`。论文只承诺开源**模型权重**，没有承诺开源训练数据。
> 所以训练侧无法精确复现。
>
> 对比 M3：它的微调数据 100% 公开（17 个点名数据集），只有预训练的 MTP 与
> MultiLongDoc 合成数据拿不到（见 `01-bge-m3.md`）。

> Qwen3-Reranker **不做阶段一的弱监督预训练**，直接从微调阶段开始 —— 这是
> 它与 Embedding 模型训练上的唯一本质区别。

> 与 M3 相同：T²-Ranking 是**微调**数据之一，不是评测集。

## 评测集与指标

主评测是 **MMTEB**，共 **216 个任务**：

| 子集 | 任务数 | 论文对应表 | 内容 |
|---|---|---|---|
| MTEB (Multilingual) | 131 | `tables/mteb_multilingual.tex` | 多语检索/分类/STS/聚类等 |
| MTEB (English, v2) | 41 | `tables/mteb_en.tex` | 英文 |
| CMTEB | 32 | `tables/mteb_zh.tex` | 中文 |
| MTEB (Code) | 12 | `tables/mteb_code.tex` | 代码检索 |

指标按任务类型取该任务主指标，检索类统一 **nDCG@10**。

> ⚠️ **论文没有列出 216 个任务的名字**。附录里的 4 张表只给"模型 × 子集均值"，
> 没有单任务清单。完整任务清单在 MMTEB 官方仓库
> （`github.com/embeddings-benchmark/mteb`）里，用 `mteb` 库按版本加载即可 ——
> **这里必须以官方清单为准，论文的 216 是某个特定版本的数字**。

### MTEB(English, v2) 论文得分（表 3）

列是**任务类型**，不是单个任务：

| 模型 | Mean(Task) | Mean(Type) | Classification | Clustering | PairClass | Rerank | Retrieval | STS | Summ |
|---|---|---|---|---|---|---|---|---|---|
| stella_en_1.5B_v5 | 69.43 | 65.32 | 89.38 | 57.06 | 88.02 | 50.19 | 52.42 | 83.27 | 36.91 |
| gte-Qwen2-7B-instruct | 70.72 | 65.77 | 88.52 | 58.97 | 85.90 | 50.47 | 58.09 | 82.69 | 35.74 |
| multilingual-e5-large-instruct | 65.53 | 61.21 | 75.54 | 49.89 | 86.24 | 48.74 | 53.47 | 84.72 | 29.89 |
| NV-Embed-v2 | 69.81 | 65.00 | 87.19 | 47.66 | 88.69 | 49.61 | 62.84 | 83.82 | 35.21 |
| GritLM-7B | 67.07 | 63.22 | 81.25 | 50.82 | 87.29 | 49.59 | 54.95 | 83.03 | 35.65 |
| **Qwen3-Embedding-0.6B** | **70.70** | **64.88** | **85.76** | **54.05** | **84.37** | **48.18** | **61.83** | **86.57** | **33.43** |
| Qwen3-Embedding-4B | 74.60 | 68.09 | 89.84 | 57.51 | 87.01 | 50.76 | 68.46 | 88.72 | 34.39 |
| Qwen3-Embedding-8B | 75.22 | 68.70 | 90.43 | 58.57 | 87.52 | 51.56 | 69.44 | 88.58 | 34.83 |

MTEB 的汇总口径有 **Mean(Task)** 和 **Mean(Type)** 两种，复现时必须说明用哪个
（论文主表用的是 Mean(Task)）。


## 对齐方案

1. 用 `mteb` 库（`github.com/embeddings-benchmark/mteb`）按官方任务清单跑
2. 需要按各任务的**官方 corpus 全集**检索，不是固定候选数
3. 报告 216 任务的分类均值（MTEB 的 `mean` 聚合口径）
4. 我们的 T²-Ranking 16 候选用法**与 MMTEB 不可比**，只能当独立实验
