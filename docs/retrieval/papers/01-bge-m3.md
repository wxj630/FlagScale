# 01 · BGE-M3（M3-Embedding）

原文：`Papers/M3-Embedding/main.tex`，ACL Findings 2024，§3–4、表 8。
模型权重：`BAAI/bge-m3`。

## 训练阶段：实际是**三阶段**，不是两阶段

论文 §4 原文：*"The training process constitutes a **multi-stage workflow**"*
（`main.tex:348`）。附录 §A 把三段拆开写了。

### 三个阶段的区别（这是核心，不只是数据量不同）

| | **阶段 0 · RetroMAE** | **阶段 1 · 无监督预训练** | **阶段 2 · 统一微调** |
|---|---|---|---|
| **起点** | XLM-RoBERTa-large | bge-m3-retromae | bge-m3-unsupervised |
| **训练目标** | MAE 式重建（非检索）| **只有 dense**（单一向量）| **dense + sparse + multi-vector 三路** |
| **损失** | RetroMAE 的 Encoder/Decoder 重建 + 对比 | basic contrastive | `L_final = (L + L') / 2`，L' 是三路加权 |
| **知识蒸馏** | ❌ | ❌ | ✅ **自知识蒸馏**（三路融合当教师信号）|
| **硬负例** | — | in-batch | **每 query 采 7 个（ANCE）** |
| **数据** | Pile + Wudao + mC4，**1.84 亿条，105 语言** | **1.2B** 无监督，194 语言 | **1.6M** 有监督（17 集）+ 合成 |
| **序列长度** | **扩展到 8192** | query 512 / passage 8192 | 同上 |
| **lr** | **7e-5** | **5e-5**（warmup 0.1，wd 0.01）| 先 6,000 步 warm-up 再统一训练 |
| **batch** | 32 × grad_accum 16 | 按长度分档（见论文表）| 按长度分档 |
| **步数** | **20,000** | **25,000** | 6,000 warm-up + 统一训练 |
| **算力** | 32×A100(40G) | **96×A800(80G)** | **24×A800(80G)** |

> 引用位置：阶段 0 见 `main.tex:826-834`，阶段 1 见 `:836`，阶段 2 见 `:843-845`。

**一句话**：阶段 0 用 MAE 把序列长度撑到 8192 并打好语言基础；阶段 1 只教
"把语义压成一个向量、按对比学习排序"；阶段 2 才扩展成"一个模型三种检索方式"，
并用三路集成结果反过来蒸馏自己。

### 三个阶段各自贡献多少（论文表 3，MIRACL dev nDCG@10）

| 训练方式 | MIRACL nDCG@10 |
|---|---|
| Fine-tune（直接从 XLM-R） | 60.5 |
| **+ RetroMAE** 预训练 | **66.1（+5.6）** |
| **+ 无监督预训练** | **69.2（再 +3.1）** |

> **RetroMAE 那一步贡献 +5.6 点，比整个无监督预训练（+3.1）还大。**
> 论文原话：*"RetroMAE can significantly improve the retrieval performance"*
> （`main.tex:670`）。

### ✅ 每个阶段的权重都开源了

**这不是"只发最终模型"——BAAI 把三个中间阶段的 checkpoint 都放出来了：**

| 阶段 | HF 仓库 | model card 原文 | 许可 |
|---|---|---|---|
| 0→1 | **`BAAI/bge-m3-retromae`** | *"extend the max_length of xlm-roberta to 8192 and further pretrained via retromae"* | MIT |
| 1→2 | **`BAAI/bge-m3-unsupervised`** | *"contrastive learning from bge-m3-retromae"* | MIT |
| 2→3 | `BAAI/bge-m3` | *"unified fine-tuning (dense, sparse, and colbert) from bge-m3-unsupervised"* | MIT |

（2026-11-12 用 HF API 核实：三个仓库均可访问，`bge-m3-retromae` 含
`pytorch_model.bin` + tokenizer，303 downloads。）

**RetroMAE 方法本身也开源**：`github.com/staoxiao/RetroMAE`（Apache-2.0，276 stars），
论文里的做法是**用它继续训练 XLM-RoBERTa-large**，不是从零训练。

### 我们能复现到哪一步

| 阶段 | 需要什么 | 我们能做吗 |
|---|---|---|
| **阶段 0（RetroMAE）** | 1.84 亿文本（Pile/Wudao/mC4，公开）+ 32×A100 | ⚠️ 数据公开、方法开源，但算力远超我们（32 卡 × 20,000 步）|
| **阶段 1（无监督）** | 1.2B 语料（**MTP 是内部数据**）+ 96×A800 | ❌ 数据不可得，算力也不够 |
| **阶段 2（微调）** | 1.6M 标注（**17 个集全部公开**）+ 合成 | ⚠️ 标注可得，MultiLongDoc 合成数据未发布 |

> **但阶段 0/1 的成品权重可以免费下载**（见上表）。所以我们不需要重训——
> 从 `bge-m3-unsupervised` 或 `bge-m3-retromae` 出发，只做我们要验证的那一段。

## 三个阶段的完整数据集清单

**注意：论文表 8（`tab:data`）只有两栏（Unsupervised / Fine-tuning），
RetroMAE 阶段的数据不在表 8 里**，写在附录 §A 正文（`main.tex:830`）。
所以三个阶段的清单要从两处拼起来。

### 阶段 0 · RetroMAE 继续预训练 —— 1.84 亿条，105 语言

| 来源 | 规模 | 可得性 |
|---|---|---|
| **Pile** | 三者合计采样 **1.84 亿条文本**（论文未给各自占比）| ✅ 公开 |
| **Wudao** | 同上 | ✅ 公开 |
| **mC4** | 同上 | ✅ 公开 |

> 原文：*"The data comprises Pile, Wudao, and mC4 datasets. We sampled a total of
> 184 million text samples from these sources, covering 105 languages."*
>
> 这一阶段只做 MAE 式重建 + 把 max_length 从 512 扩到 8192，**不做检索训练**。
> 论文刻意把它与下面 1.2B 的检索预训分开算（消融见表 3）。

### 阶段 1 · 无监督预训练 —— 1.2B，194 语言（论文表 8）

| 来源 | 语言 | 规模 | 可得性 |
|---|---|---|---|
| **MTP**（BGE 自建，title-body 等结构抽取）| EN, ZH | 291.1M | ❌ **内部数据，未公开** |
| **S2ORC、Wikipedia** | EN | 48.3M | ⚠️ 语料公开，清洗流程未公开 |
| **xP3、mC4、CC-News** | 多语 | 488.4M | ⚠️ 同上 |
| **NLLB、CCMatrix**（平行句）| 跨语 | 391.3M | ✅ 公开 |
| **CodeSearchNet** | 文本-代码 | 344.1K | ✅ 公开 |
| **合计** | **194 语言，2,655 跨语言对应** | **1.2B** | ❌ **无法完整重建** |

> 抽取方式是"rich-semantic structures"，即 title-body / title-abstract /
> instruction-output 这类结构对（`main.tex:272`）。
>
> **长文本处理**：把全文切成 3 段、随机打乱顺序再拼接，**以 0.2% 概率**施加。
> 动机是新闻类文本的开头常有摘要句，模型会只依赖开头建立相关性
> （`main.tex:789-790`）。

#### 数据长什么样：**不是三元组，是 (text_a, text_b) 对**

这是理解这一阶段最关键的一点。1.2B 的统计单位是**「text pair」**（1.2 billion
**text pairs**），每个样本由来源文档内部的**结构化关系**自动生成：

| 来源 | 取哪两个字段当 (a, b) | 例子 |
|---|---|---|
| **MTP** | title ↔ body | 文章标题 / 正文 |
| **S2ORC** | title ↔ abstract | 论文标题 / 摘要 |
| **xP3** | instruction ↔ output | 任务指令 / 回答 |
| **Wikipedia** | 段落结构（title-body 类）| 条目标题 / 首段 |
| **NLLB / CCMatrix** | 源句 ↔ 译文 | 平行语料，天然成对 |
| **CodeSearchNet** | 文档字符串 ↔ 函数体 | 文本-代码对 |

**没有人工标注的「正例/负例」标签。** 正例就是这对结构关系本身；
负例全部来自 **in-batch**——同一个 batch 里其它对的 text_b。
这就是为什么论文称它为 "unsupervised"：**监督信号来自文档自身的结构，不是人工标注。**

> 对比阶段 2：那里的正例是标注出来的 (query, relevant passage)，
> 还有 ANCE 挖的 7 个 hard negative。**阶段 1 完全没有 negative mining。**

#### 语言分布（论文图 3 左，直接读图）

| 语言 | 占比 |
|---|---|
| **English** | **43.9%** |
| **Chinese** | **20.5%** |
| Spanish | 6.1% |
| German | 4.0% |
| Russian | 3.0% |
| Italian | 2.9% |
| French | 2.7% |
| Arabic | 1.8% |
| Turkish | 1.8% |
| Portuguese | 1.7% |
| Hindi | 1.1% |
| **183 Others** | **10.4%** |

> 中英合计 **64.4%**，剩下 192 种语言分 35.6%。这就是为什么 M3 在低资源语言上
> 比基线稳（论文 §5.2 提到 ar、km、he 等语种基线崩了而 M3 稳定）。

#### 长度分布（论文图 3 右，直接读图）

| 序列长度区间 | 占比 |
|---|---|
| **0-1000** | **63.9%** |
| 1000-2000 | 25.7% |
| 2000-3000 | 5.9% |
| 3000-4000 | 2.0% |
| 4000-5000 | 0.9% |
| >5000 | 1.6% |

> **89.6% 的样本在 2000 token 以内**，但**有 4.5% 超过 3000**。
> 这正是论文要扩到 8192 的原因：长文数据虽然少，但必须支持。
> 也是"按长度分档设 batch size"（论文表 `tab:batch_size_dict`）的动机——
> 短样本用大 batch 提吞吐，长样本用小 batch 防 OOM。

### 阶段 2 · 统一微调 —— 1.6M 有监督 + 41.4K 合成（论文表 8 + §3.2）

论文正文 §3.2 **逐个点名**，清单完整、**无 `.etc`**：

| 来源 | 语言 | 规模 | 可得性 |
|---|---|---|---|
| **HotpotQA、TriviaQA、NQ、MS MARCO、COLIEE、PubMedQA、SQuAD、SimCSE-NLI** | EN（8 个）| 1.1M | ✅ 公开 |
| **DuReader、mMARCO-ZH、T²-Ranking、LawGPT、CMedQAv2、NLI-zh、LeCaRDv2** | ZH（7 个）| 386.6K | ✅ 公开 |
| **Mr. TyDi、MIRACL** | 多语（2 个）| 88.9K | ✅ 公开 |
| **MultiLongDoc**（合成）| 多语 | 41.4K | ❌ **未发布**，只给构造方法 |

**8 + 7 + 2 = 17 个标注数据集全部公开**，唯一缺的是合成数据。

> **MultiLongDoc 的构造方法**（`main.tex:276`）：从 **Wikipedia / Wudao / mC4**
> 采样长文 → 随机选段落 → 用 **GPT-3.5 生成问题** → (生成的问题, 文章) 作为一个训练对。
> 目的是补长文档检索数据的不足。
>
> ⚠️ **注意 MultiLongDoc 的数据源和阶段 0/1 有重叠**（都用 Wikipedia/Wudao/mC4），
> 但用途不同：阶段 0/1 是重建/对比学习，这里是生成问答对。


> 论文脚注称 "model, code, and data is publicly available"，实际发布的是
> 微调数据 `Shitao/bge-m3-data`；**预训练的 MTP 与合成数据不在其中**。

> **关键**：T²-Ranking 是 M3 的**微调**数据之一，不是评测集。
> 我们把它当训练+评测集用，与论文用法不同。

## 评测集与指标（共 4 个基准）

| 任务 | 数据集 | 主指标 | 辅助指标 | 规模 / 语言 |
|---|---|---|---|---|
| 多语检索 | **MIRACL dev** | **nDCG@10** | **Recall@100**（附录） | 18 语言，同语言 query-passage |
| 跨语检索 | **MKQA** | **Recall@100** | **Recall@20**（附录） | 25 非英语 query → 英文 Wikipedia（BEIR 处理版） |
| 长文检索 | **MLDR** test | **nDCG@10** | — | 12 语言；corpus 493,709 篇，test query 3,800 |
| 长文检索 | **NarrativeQA** | **nDCG@10** | — | 仅英文 |

**原文逐字**（`Papers/M3-Embedding/main.tex`）：

- MIRACL（`:434`）：*"we use **nDCG@10** as the primary evaluation metric
  (**Recall@100 is also measured and reported** in Appendix)"*
- MKQA（`:559`）：*"we report **Recall@100** as the primary metric
  (**Recall@20 is reported as an auxiliary metric** in the Appendix)"*
- MLDR（`:551`）与 NarrativeQA（`:613`）的表题写明 *"(measured by nDCG@10)"*

> ⚠️ **BGE-M3 的主指标是两项，不是只有 nDCG@10**：检索质量看 **nDCG@10**
> （MIRACL / MLDR / NarrativeQA），跨语召回看 **Recall@100**（MKQA）。
> 附录另有 MIRACL 的 Recall@100（`:1033`）与 MKQA 的 Recall@20（`:1083`）两张表。

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
