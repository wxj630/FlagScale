# 01 · BGE-M3（M3-Embedding）

原文：`Papers/M3-Embedding/main.tex`，ACL Findings 2024，§3–4、表 8。
模型权重：`BAAI/bge-m3`。

## 训练数据（论文表 8，分两阶段）

### 阶段一：预训练（Unsupervised，1.2B）

| 来源 | 语言 | 规模 | 可得性 |
|---|---|---|---|
| MTP（BGE 自建） | EN, ZH | 291.1M | ❌ **内部数据，未公开** |
| S2ORC、Wikipedia | EN | 48.3M | ⚠️ 语料公开，清洗流程未公开 |
| xP3、mC4、CC-News | 多语 | 488.4M | ⚠️ 同上 |
| NLLB、CCMatrix（平行句） | 跨语 | 391.3M | ✅ 公开 |
| CodeSearchNet | 文本-代码 | 344.1K | ✅ 公开 |
| **合计** | **194 语言，2655 跨语言对应** | **1.2B** | ❌ **无法完整重建** |

长文本处理：分成 3 段随机打乱顺序再拼接（概率 0.2%），避免模型只依赖开头摘要句。

### 阶段二：微调（Fine-tuning，1.6M）

论文正文（§3.2）**逐个点名**，清单完整、**全部公开**：

| 来源 | 语言 | 规模 | 可得性 |
|---|---|---|---|
| HotpotQA、TriviaQA、NQ、MS MARCO、COLIEE、PubMedQA、SQuAD、SimCSE-NLI | EN | 1.1M | ✅ 公开 |
| DuReader、mMARCO-ZH、**T²-Ranking**、LawGPT、CMedQAv2、NLI-zh、LeCaRDv2 | ZH | 386.6K | ✅ 公开 |
| Mr. TyDi、MIRACL | 多语 | 88.9K | ✅ 公开 |
| MultiLongDoc（合成） | 多语 | 41.4K | ❌ **GPT-3.5 合成，未发布** |

**英文 8 + 中文 7 + 多语 2 = 17 个数据集，无 `.etc`**，全部可下载。
唯一缺的是 MultiLongDoc 合成数据（论文只给 prompt 与规格表）。

> 论文脚注称 "model, code, and data is publicly available"，实际发布的是
> 微调数据 `Shitao/bge-m3-data`；**预训练的 MTP 与合成数据不在其中**。

> **关键**：T²-Ranking 是 M3 的**微调**数据之一，不是评测集。
> 我们把它当训练+评测集用，与论文用法不同。

## 评测集与指标（共 4 个基准）

| 任务 | 数据集 | 主指标 | 辅助指标 | 规模 / 语言 |
|---|---|---|---|---|
| 多语检索 | **MIRACL dev** | **nDCG@10** | Recall@100 | 18 语言，同语言 query-passage |
| 跨语检索 | **MKQA** | **Recall@100** | Recall@20 | 25 非英语 query → 英文 Wikipedia（BEIR 处理版） |
| 长文检索 | **MLDR** test | **nDCG@10** | — | 12 语言；corpus 493,709 篇，test query 3,800 |
| 长文检索 | **NarrativeQA** | **nDCG@10** | — | 仅英文 |

**MIRACL 的 18 语言**（论文表 7 的列）：
ar、bn、en、es、fa、fi、fr、hi、id、ja、ko、ru、sw、te、th、zh、de、yo。

**MKQA 的 25 种语言/地区**（论文表 8 的行）：
ar、da、de、es、fi、fr、he、hu、it、ja、km、ko、ms、nl、no、pl、pt、ru、sv、th、
tr、vi、**zh_cn、zh_hk、zh_tw**。

MKQA 的各方式 Recall@100 平均值（论文表 8）：

| 方式 | Avg Recall@100 |
|---|---|
| BM25 | 39.9 |
| mDPR | 60.6 |
| mContriever | 67.9 |
| mE5-large | 70.9 |
| E5-mistral-7b | 70.1 |
| M3 Dense | 75.1 |
| **M3 All** | **75.5** |

各检索方式的 nDCG@10 平均值（论文表 7，供对照）：

| 方式 | Avg | 说明 |
|---|---|---|
| BM25 | 31.9 | 词法基线 |
| mDPR | 41.8 | 稠密基线 |
| mContriever | 43.1 | 稠密基线 |
| mE5-large | 66.6 | 稠密基线 |
| E5-mistral-7b | 63.4 | 稠密基线 |
| M3 Dense | 69.2 | **我们对应的方式** |
| M3 Sparse | 53.9 | |
| M3 Multi-vec | 70.5 | 只作 reranker |
| M3 Dense+Sparse | 70.4 | w=1/0.3 |
| **M3 All** | **71.5** | w=1/0.3/1，论文最好成绩 |

### MLDR 的构成（论文表 9，12 语言，完整）

| 语言 | 来源 | #train | #dev | #test | #corpus | 文档平均长度 |
|---|---|---|---|---|---|---|
| ar | Wikipedia | 1,817 | 200 | 200 | 7,607 | 9,428 |
| de | Wikipedia, mC4 | 1,847 | 200 | 200 | 10,000 | 9,039 |
| en | Wikipedia | 10,000 | 200 | 800 | 200,000 | 3,308 |
| es | Wikipedia, mC4 | 2,254 | 200 | 200 | 9,551 | 8,771 |
| fr | Wikipedia | 1,608 | 200 | 200 | 10,000 | 9,659 |
| hi | Wikipedia | 1,618 | 200 | 200 | 3,806 | 5,555 |
| it | Wikipedia | 2,151 | 200 | 200 | 10,000 | 9,195 |
| ja | Wikipedia | 2,262 | 200 | 200 | 10,000 | 9,297 |
| ko | Wikipedia | 2,198 | 200 | 200 | 6,176 | 7,832 |
| pt | Wikipedia | 1,845 | 200 | 200 | 6,569 | 7,922 |
| ru | Wikipedia | 1,864 | 200 | 200 | 10,000 | 9,723 |
| th | mC4 | 1,970 | 200 | 200 | 10,000 | 8,089 |
| zh | Wikipedia, Wudao | 10,000 | 200 | 800 | 200,000 | 4,249 |
| **合计** | Wikipedia / Wudao / mC4 | **41,434** | **2,600** | **3,800** | **493,709** | **4,737** |

MLDR 是 M3 论文**自己构造**的数据集（GPT-3.5 从长文生成问题），
与训练用的 MultiLongDoc 同源但不同 split。

MLDR 是 M3 论文**自己构造**的数据集（用 GPT-3.5 从长文中生成问题），
与 MultiLongDoc 训练集同源但不同 split。

### 检索方式（论文报 5 种，我们只做 Dense）

| 方式 | 做法 |
|---|---|
| Dense | 编码全语料 → Faiss 建索引 → top-1000 |
| Sparse | 词权重 → Lucene 索引 → top-1000 |
| Multi-vec | **只作 reranker**，重排 dense 的 top-200（成本高） |
| Dense+Sparse | 合并两个 top-1000 重排，w₁=1, w₂=0.3 |
| All | dense top-200 上用三路融合，w₁=1, w₂=0.3, w₃=1 |

## 对齐方案（若要与论文数值比较）

1. 下载 MIRACL（HF `miracl/miracl`）与 MKQA（`apple-aiml-research/ml-mkqa`）
2. **全集检索**，不能只取 16 候选
3. 主报 nDCG@10；长文另报 MLDR nDCG@10
4. 我们只做 Dense，论文的 All（1/0.3/1）是它的最好成绩，对比时要标注
