# 03 · Qwen3-Reranker-8B

原文：`Papers/Qwen3-Embedding/contents/evaluation.tex` §Evaluation Settings、
`tables/rerank_all.tex`（表 4）。模型权重：`Qwen/Qwen3-Reranker-8B`。

## 训练数据

**没有预训练阶段**（与 Embedding 模型的关键区别），直接用与 Qwen3-Embedding
相同的**监督微调**数据。论文表 5 原文（结尾是 `.etc`，未给完整清单）：

```latex
MS MARCO, NQ, HotpotQA, NLI, Dureader, T²-Ranking,
SimCLUE, MIRACL, MLDR, Mr.TyDi, Multi-CPR, CodeSearchNet .etc
+ High-quality Synthetic Data
```

| 来源 | 规模 | 可得性 |
|---|---|---|
| 标注数据（上列 12 个 + `.etc` 未公开部分） | ~7M | ⚠️ 12 个公开，其余未知 |
| 高质量合成数据 | ~12M | ❌ **未发布** |

损失函数用 reranking 专用的 cross-entropy（不是 Embedding 的对比损失）。

> ⚠️ 训练侧**无法精确复现**：合成数据占 63%（12M / 19M）且未发布，
> 清单还有 `.etc`。与 Embedding 模型是同一个问题。

## 评测集与指标

**6 个基准，统一 nDCG@10**：

| 基准 | 类别 | 内容 |
|---|---|---|
| MTEB-R | 英文 | MTEB (eng, v2) 的检索子集 |
| CMTEB-R | 中文 | MTEB (cmn, v1) 的检索子集 |
| MMTEB-R | 多语 | MMTEB 的检索子集 |
| MLDR | 长文 | 多语长文档检索 |
| MTEB-Code | 代码 | 纯代码检索 |
| FollowIR | 复杂指令 | 带指令的检索 |

## 关键：候选来源

> 论文原文：*"We utilize the Qwen3-Embedding-0.6B model to retrieve the **top-100**
> candidates and then apply different reranking models for further refinement.
> This approach ensures a fair evaluation of the reranking models."*

**所有 reranker 都在同一批 top-100 候选上重排**，候选由
`Qwen3-Embedding-0.6B` 提供。这是论文为了"公平比较 reranker"专门设计的协议。

## 论文数值（表 4，8B）

| MTEB-R | CMTEB-R | MMTEB-R | MLDR | MTEB-Code | FollowIR |
|---|---|---|---|---|---|
| 69.02 | **77.45** | **72.94** | **70.19** | **81.22** | 8.05 |

## 对齐方案

必须复现两步，缺一不可：

1. **先用 `Qwen3-Embedding-0.6B` 检索 top-100**（不是我们的 16 候选）
2. 再用 reranker 重排这 100 个，报 nDCG@10

我们现在用 T²-Ranking 的 16 候选（1 正 + 15 负），**与论文协议不同**：
- 候选数不同（16 vs 100）
- 候选来源不同（数据集给定 vs embedding 模型检索）
- 指标不同（R@1/5/10 + MRR vs nDCG@10）
