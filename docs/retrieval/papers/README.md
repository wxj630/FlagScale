# papers/ — 论文对照

三个模型的**数据集、评测集、主指标**与原论文的逐项对照，以及我们实现与论文要求的差异。

原文依据：`Papers/M3-Embedding/main.tex`、`Papers/Qwen3-Embedding/contents/evaluation.tex`、
`Papers/CLIP/clip_paper.tex`（LaTeX 源，最准）。

| 文档 | 用途 |
|---|---|
| [01-bge-m3](01-bge-m3.md) | 4 个基准（MIRACL/MKQA/MLDR/NarrativeQA）与主指标 |
| [02-qwen3-embedding](02-qwen3-embedding.md) | MMTEB 216 任务 |
| [03-qwen3-reranker](03-qwen3-reranker.md) | 6 个基准；**注意论文没逐项声明指标**，FollowIR 是 p-MRR |
| [04-clip](04-clip.md) | Flickr30k / MSCOCO 5k；**只有 R@1/5/10，无 nDCG** |
| [05-gaps](05-gaps.md) | 我们现在跑的 vs 论文要求 |

> **关键**：三者的主指标不同，汇报时**不取并集**。BGE 是 nDCG@10 + Recall@100，
> Reranker 是 nDCG@10（FollowIR 用 p-MRR），CLIP 只有 R@1/5/10。
