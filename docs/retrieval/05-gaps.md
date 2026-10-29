# 05 · 当前实现 vs 论文要求

逐项列出我们的实验与论文的差距。**结论：现在的结果只能用于平台间横向比较
和「训练前后」的增益判断，不能直接对标论文数值。**

## 差异 0：训练侧到底能不能复现（先看这条）

判断标准不能只看"论文有没有给清单"，还要看**数据我们拿不拿得到**。按这两条
重新审视，结论是：**三个模型的训练侧都无法精确复现**，区别只是差多少。

### BGE-M3

| 阶段 | 数据 | 规模 | 公开？ |
|---|---|---|---|
| 预训练 | MTP（BGE 自建） | 291.1M | ❌ **内部数据**，未公开 |
| 预训练 | S2ORC、Wikipedia、xP3、mC4、CC-News | 536.7M | ✅ 源语料公开，**但过滤/清洗流程未公开**，重建结果不等价 |
| 预训练 | NLLB、CCMatrix 平行句 | 391.3M | ✅ 公开 |
| 预训练 | CodeSearchNet | 344.1K | ✅ 公开 |
| 微调 | 英/中/多语 17 个数据集 | 1.6M | ✅ **全部公开**（清单完整，逐个点名） |
| 微调 | MultiLongDoc | 41.4K | ❌ **GPT-3.5 合成**，仅给 prompt 与规格，未发布数据文件 |

**判定：❌ 不能。** 缺 MTP 291.1M（预训练的 24%）与 MultiLongDoc 全部。
论文号称"data publicly available"（脚注指向 FlagOpen/FlagEmbedding），但
那指的是微调数据 `Shitao/bge-m3-data`；预训练的 MTP 与合成数据不在其中。

**这就是我们当前的处境**：拿不到 1.2B 预训练对，只能从已发布的 bge-m3
checkpoint 出发，用公开的 T²-Ranking 做微调 —— 与论文的多阶段训练不是一回事。

### Qwen3-Embedding / Reranker

| 阶段 | 数据 | 规模 | 公开？ |
|---|---|---|---|
| 弱监督预训练 | Qwen3-32B 合成 | ~150M | ❌ **合成数据未发布**，只给 prompt 模板 |
| 微调-标注 | 12 个点名数据集 + **`.etc` 未公开部分** | ~7M | ⚠️ 点名的 12 个公开，其余未知 |
| 微调-合成 | 从 150M 里 cos>0.7 筛出 | ~12M | ❌ 未发布 |

**判定：❌ 不能，且比 M3 更差。** 两个阶段的合成数据（占微调数据的 63%：
12M / 19M）完全拿不到，且清单本身就有 `.etc`。

论文只承诺开源**模型权重**（"making them available to the community"），
**没有承诺开源训练数据**。

### CLIP

| 阶段 | 数据 | 规模 | 公开？ |
|---|---|---|---|
| 论文预训练 | WIT | 4 亿对 | ❌ **明确未公开**（自建爬取） |
| **我们的微调** | MSCOCO Captions 2017 train | 21 shard / 19 GB，3 epoch 全参 | ✅ 公开 |

**判定：❌ 不能复现论文的训练。** 论文只说 "release our code and pre-trained
model weights"，数据从未发布。

我们所做的是论文里**不存在的一步**：拿 COCO train 全参微调。这带来一个
必须写进结论的偏差 —— **我们训练时见过 COCO，论文是零样本**。所以微调后
R@1 高于论文数值是预期内的，不代表模型更强。

### 汇总

| 模型 | 数据可得性 | 训练侧能否精确复现 |
|---|---|---|
| BGE-M3 | 预训练缺 MTP(291M) + 合成 RL；微调 100% 公开 | ❌ 不能（缺预训练 24%） |
| Qwen3-Embedding | 缺全部合成数据，占微调 63%；清单有 `.etc` | ❌ 不能 |
| Qwen3-Reranker | 同 Embedding | ❌ 不能 |
| CLIP | WIT 4 亿对完全未公开；微调用 COCO train（论文没有这一步） | ❌ 不能 |

**后果**：训练侧不可能是等价复现。我们实际做的是
**"从官方预训练 checkpoint 出发，在公开数据上做增量微调"**，
这本身是个合理的实验，但报告时必须写成这个口径，
**不能说"复现了论文的训练"**。

能对标论文的只有**评测侧**（见下面差异 1-4），而那也需要换数据集、改候选池、
改指标。

## 总览

| 模型 | 数据集 | 候选池 | 指标 | 能对标论文？ |
|---|---|---|---|---|
| BGE-M3 | T²-Ranking（论文的**训练**集） | 16（1 正 + 15 负） | R@1/5/10 + MRR | ❌ |
| Qwen3-Embedding | 同 BGE-M3 | 16 | 同 | ❌ |
| Qwen3-Reranker | 同 BGE-M3 | 16 | 同 | ❌ 论文要 top-100 候选 |
| CLIP | MSCOCO 5k test | 5000 全库 | R@1/5/10 双向 | ⚠️ 接近，但缺 `a photo of`，且论文是 zero-shot |

## 差异 1：评测集不是论文的评测集

| 模型 | 论文评测集 | 我们用的 |
|---|---|---|
| BGE-M3 | MIRACL / MKQA / MLDR / NarrativeQA | **T²-Ranking** |
| Qwen3-Embedding | MMTEB 216 任务 | **T²-Ranking** |
| Qwen3-Reranker | MTEB-R/CMTEB-R/MMTEB-R/MLDR/MTEB-Code/FollowIR | **T²-Ranking** |

T²-Ranking 是三篇论文的**训练**数据之一（M3 表 8 里中文监督 386.6K 的一部分），
用它做评测等于在论文的训练分布上测，**和论文的评测集没有交集**。

## 差异 2：候选池规模

| | 论文 | 我们 |
|---|---|---|
| 文本三模型 | 全语料检索（MIRACL 全 corpus / MTEB 官方 corpus） | 每 query **16 候选** |
| Reranker | **top-100**（由 Qwen3-Embedding-0.6B 检索） | 16 候选（数据集给定） |
| CLIP | 5k test 全库 | **5000 全库** ✅ 一致 |

16 候选的 R@1 随机基线是 1/16 = 6.25%，5k 全库是 1/5000 = 0.02%。
**两者难度差两个数量级**，所以小候选池上的 R@1 天然偏高、涨幅天然偏小
（天花板效应）。

## 差异 3：指标（已修复一部分）

| 论文主指标 | 我们原来报的 | 现在报的 |
|---|---|---|
| **nDCG@10**（M3 的 MIRACL/MLDR/NarrativeQA；Qwen3 全部 6 个基准） | ❌ **没有** | ✅ **已加** |
| Recall@100（M3 的 MKQA） | ❌ | ⚠️ 16 候选池下 recall@20 = recall@16，无法到 100 |
| Recall@20 / nDCG@20（LightGCN 类基线口径） | ❌ | ✅ **已加** |
| R@1/5/10（CLIP） | ✅ | ✅ 不变（与论文表 13 一致） |
| MRR / mean rank | ✅ | ✅ 保留 |

### 已做的修改（2026-10-29）

`examples/retrieval/evaluate_retrieval.py` 的 `rank_metrics()` 现在同时输出：

```python
recall@{1,5,10,20}   # 命中率
ndcg@{1,5,10,20}     # 论文主指标
hit@{1,5,10,20}      # = recall@k，供基线按 HR@k 口径对比
mrr / mean_rank
```

nDCG 实现说明：本任务每个 query 只有 1 个相关文档（ground truth 是候选 0），
所以 `DCG = 1/log2(rank+1)`、`IDCG = 1`，nDCG@k 就是该增益在 k 处截断后的均值。
**这个简化只在一相关文档时成立**，多相关文档的 nDCG 需要改用完整 DCG 公式。

### 仍然缺的

| 缺口 | 影响 | 补法 |
|---|---|---|
| **Recall@100 / MAP** | M3 的 MKQA 主指标是 Recall@100，16 候选池做不到 | 需全语料检索（改评测协议，不只是加指标） |
| **11-point mAP** | CLIP 的 Pascal VOC 用这个 | 我们不做分类任务，不需要 |
| **mean per class** | CLIP 的 FGVC Aircraft / Oxford Pets / Flowers/Caltech 用这个 | 同上，分类任务才需要 |
| **ROC AUC** | CLIP 的 Hateful Memes | 同上 |
| **mean(top1, top5)** | CLIP 的 Kinetics700 | 同上 |
| **MTEB 的任务类型均值**（Mean(Task) vs Mean(Type)） | Qwen3-Embedding 主表口径 | 需跑完整 MMTEB，用 `mteb` 库自带聚合 |

**结论**：文本三模型的指标**已经能覆盖论文主指标**（nDCG@k 是核心，已补）。
真正的缺口不是指标，而是**候选池**（16 vs 全语料 / top-100），
那个改起来比加指标大得多。

## 差异 4：CLIP 的两个细节

1. 论文评测时文本侧加 **`a photo of`** 前缀（涨 1-2 点 R@1），我们没加
2. 论文是 **zero-shot**，我们是微调后 —— 数值上完全不同的两件事

## 要达到论文口径需要做什么

按优先级：

1. **换评测集**：BGE-M3 / Qwen3-Embedding 用 MIRACL + MMTEB，
   Reranker 用 MTEB-R/CMTEB-R/MMTEB-R/MLDR/MTEB-Code/FollowIR
2. **改候选池**：文本模型做全语料检索；reranker 先用
   `Qwen3-Embedding-0.6B` 检索 top-100 再重排
3. **加 nDCG@10**：`tools/recompute_metrics.py` 已支持 `--ndcg`，
   但需要全语料打分才能算出有意义的 nDCG
4. **CLIP 加 `a photo of` 前缀**

前两项工作量最大（要下 MIRACL / MMTEB 并做全库检索），但是唯一能对标论文的路径。
在当前 T²-Ranking 口径下，跨平台横向比较和训练增益判断仍然有效。
