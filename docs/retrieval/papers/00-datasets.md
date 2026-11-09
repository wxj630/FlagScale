# 数据集目录（三模型 × 预训练/微调/评测）

**一页看全三个模型用了什么数据、能不能拿到、我们实际用了什么。**
细节与论文原文引用在各模型文档里（[01](01-bge-m3.md) / [02](02-qwen3-embedding.md) /
[03](03-qwen3-reranker.md) / [04](04-clip.md)）。

## 一、总览

| 模型 | 预训练 | 微调 | 评测 | 能完整复现？ |
|---|---|---|---|---|
| **BGE-M3** | 1.2B（194 语言） | 1.6M（17 个数据集） | MIRACL / MKQA / MLDR / NarrativeQA | ❌ MTP 内部数据未公开 |
| **Qwen3-Embedding** | ~150M 合成对 | ~7M 标注 + 12M 合成 | MMTEB 216 任务 | ❌ 合成数据未发布 |
| **Qwen3-Reranker** | **无**（直接微调） | ~7M 标注 + 12M 合成 | 6 个 MTEB 系基准 | ❌ 合成占 63% |
| **CLIP** | WIT 4 亿图文对 | 论文无微调阶段 | Flickr30k / MSCOCO 5k | ❌ WIT 未公开 |

> **四个模型都做不到完整复现预训练。** 差距的来源不同：BGE 是内部 MTP 数据，
> 两个 Qwen3 是未发布的合成数据，CLIP 是 4 亿私有图文对。

## 二、按模型展开

### BGE-M3

**预训练 1.2B（194 语言）**

| 来源 | 规模 | 可得性 |
|---|---|---|
| MTP（BGE 自建） | 291.1M | ❌ 内部数据 |
| S2ORC、Wikipedia | 48.3M | ⚠️ 语料公开，清洗流程未公开 |
| xP3、mC4、CC-News | 488.4M | ⚠️ 同上 |
| NLLB、CCMatrix | 391.3M | ✅ 公开 |
| CodeSearchNet | 344.1K | ✅ 公开 |

**微调 1.6M（17 个数据集，论文逐个点名，无 `.etc`）**

| 语言 | 数据集 |
|---|---|
| 英文 8 | HotpotQA、TriviaQA、NQ、MS MARCO、COLIEE、PubMedQA、SQuAD、SimCSE-NLI |
| 中文 7 | DuReader、mMARCO-ZH、**T²-Ranking**、LawGPT、CMedQAv2、NLI-zh、LeCaRDv2 |
| 多语 2 | Mr. TyDi、MIRACL |
| 合成 1 | MultiLongDoc（41.4K，❌ GPT-3.5 合成未发布）|

> 实际发布的是微调数据 `Shitao/bge-m3-data`；**预训练的 MTP 与合成数据不在其中**。

### Qwen3-Embedding

**预训练 ~150M（弱监督合成对）** — ❌ 只给 prompt 模板，数据本身未发布
（Qwen3-32B 生成，可控 question_type / difficulty / length / language）

**微调 ~7M 标注 + 12M 合成** — 论文表 5 原文结尾有 `.etc`，清单不完整：

```latex
MS MARCO, NQ, HotpotQA, NLI, Dureader, T²-Ranking,
SimCLUE, MIRACL, MLDR, Mr.TyDi, Multi-CPR, CodeSearchNet .etc
+ High-quality Synthetic Data
```

### Qwen3-Reranker

**无预训练阶段**（与 Embedding 的关键区别），微调数据与 Embedding 相同：

| 来源 | 规模 | 可得性 |
|---|---|---|
| 标注（上列 12 个 + `.etc`） | ~7M | ⚠️ 12 个公开，其余未知 |
| 高质量合成 | ~12M | ❌ 未发布 |

> **合成数据占 63%（12M / 19M）**，这是 reranker 无法精确复现的主因。

### CLIP

**预训练 WIT 4 亿图文对** — ❌ 未公开（论文只给规模与筛选流程）

**微调**：论文**没有**微调阶段（表 13 是 zero-shot 结果）。我们的微调是自加的：
MSCOCO Captions 2017 train，21 shard / 19 GB，全参 3 epoch。

**评测**：Flickr30k 1k、MSCOCO 5k（双向 R@1/5/10）。
论文另有 27 个迁移评测集（附录 A.1），我们未跑。

## 三、我们实际用了什么

| 模型 | 训练数据 | 评测数据 | 与论文的差异 |
|---|---|---|---|
| BGE-M3 | **T²-Ranking**（论文的微调数据之一） | **T²-Ranking test**（16,740 query） | 论文评测集是 MIRACL/MKQA/MLDR/NarrativeQA |
| Qwen3-Reranker | **T²-Ranking** | **T²-Ranking test**（16 候选） | 论文要 top-100 候选（embedding 检索） |
| CLIP | MSCOCO Captions 2017 train | MSCOCO 5k | 论文是 zero-shot；Flickr30k 未跑 |

> ⚠️ **BGE 和 reranker 都在 T²-Ranking 上训练并评测。** 这是论文的**微调**数据，
> 不是评测集——用它当评测集，测的是同分布表现，绝对值不能与论文表比。
> 只能用于三平台横向对照（同数据同代码同口径）。

## 四、数据量的量级对照

| 模型 | 论文规模 | 我们实际 | 倍数差 |
|---|---|---|---|
| BGE-M3 预训练 | 1.2B | 未做 | — |
| BGE-M3 微调 | 1.6M | **4,602,000 texts**（23,010 步 × 200，3 epoch 累计） | 2.9× |
| Qwen3-Reranker 微调 | ~19M（标注 7M + 合成 12M） | **306,806 行/epoch**（每行 2 对 = 61.4 万对） | 论文是 ~19M **对**，量级差 31× |
| CLIP 预训练 | 400M 对 | 未做 | — |
| CLIP 微调 | 论文无 | **318,720 对**（830 步 × 3 epoch × 128） | — |

> GPU-hours 估算（[06](../analysis/06-gpu-hours.md)）基于论文规模，
> 不是我们实际跑的规模——两者差 1-2 个数量级，别混用。

## 五、能不能拿到：可执行清单

| 数据 | 链接/来源 | 状态 |
|---|---|---|
| T²-Ranking | HuggingFace（11 parquet，340,111 行） | ✅ `/flagos-retrieval-datasets/t2ranking` |
| MSCOCO Captions 2017 | cocodataset.org | ✅ `/flagos-retrieval-datasets/wds_mscoco_captions2017` |
| BGE 微调数据 | `Shitao/bge-m3-data` | ✅ 可下载 |
| MIRACL / MKQA / MLDR / NarrativeQA | 各官方页 | ✅ 可下载（需改评测协议为全语料检索）|
| MMTEB / MTEB 系 | `mteb` 库 | ✅ 可下载 |
| Flickr30k | shannon.cs.illinois.edu | ⚠️ 需申请 |
| **BGE MTP** | — | ❌ 内部 |
| **Qwen3 合成数据** | — | ❌ 未发布 |
| **MultiLongDoc** | — | ❌ 仅给 prompt |
| **CLIP WIT** | — | ❌ 未公开 |
