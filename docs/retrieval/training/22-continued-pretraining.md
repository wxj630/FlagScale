# 22 · 继续预训练数据与流程

**这不是验收那套微调实验。** 验收实验（[07](../results/07-a100.md)–[09](../results/09-ppu.md)）
是在 T²-Ranking 上微调、又在同一份 T²-Ranking 上评测，测的是**同分布拟合**。
本文讲的是**继续预训练**：拿一个已经训好的检索模型，喂一批别的语料，看能力是变好还是变坏。

两者用**同一个训练循环、同一套 loss**，差别只在数据和目标。

---

## 一、为什么"继续预训练"不是"换个数据集就行"

| | 验收微调 | 继续预训练 |
|---|---|---|
| 起点 | 预训练权重 | 预训练权重（相同）|
| 数据形态 | (query, 正例, 硬负例) 三元组 | **只有正例对**，负例来自 batch 内 |
| loss | BCE（reranker）/ 对比（embedding）| **不变**，同一个 `embedding_loss` / `clip_loss` |
| 负例来源 | 数据集给的 hard negative | **in-batch**：同 batch 的其它样本 |
| 评测集 | 训练集本身（同分布）| **必须换成另一个语料** |

**关键区别是负例。** 验收数据自带 `negative_1`，继续预训练语料没有——
所以负例只能从 batch 内部取。这不是我们发明的，BGE/Qwen3 的预训练本身
就是这么做的（in-batch negatives）。

> ⚠️ **训练和评测不能同源。** 用 T²-Ranking 训练再用 T²-Ranking 评测，
> 得到的数字是记忆量，不是能力。所以下面三个配置的 `test_path`
> **全部指向 T²-Ranking**，而 `data.path` 全部指向新数据。

---

## 二、三个数据集（全部 ModelScope 可下载，已实测）

| 模型 | 数据集 | 大小 | 字段 | 需要转换？ |
|---|---|---|---|---|
| **BGE-M3** | `C-MTEB/LCQMC` | **14 MB** | `sentence1` / `sentence2` / `score` | ❌ 直接读 |
| **CLIP** | `lmms-lab/Flickr30K` | 4.4 GB | `image` / `caption` | ❌ 直接读 |
| **Reranker** | `mteb/MSMARCO` | 3.5 GB | `corpus` / `queries` / `qrels` | ✅ 需转三元组 |

**每一行都是用 ModelScope API 实名查过的**：不存在的仓会返回
`不存在的数据集`，我们据此筛掉了 `AI-ModelScope/flickr30k`、
`modelscope/flickr8k`、`AI-ModelScope/hotpot_qa` 等一批看似合理的错误名字。

### 2.1 BGE-M3 ← LCQMC（推荐入门选它）

- 238,766 条中文问句对，**14 MB**
- 正例对示例：`我户口是外地的，可以在武汉离婚吗` / `外地户口可以在武汉离婚吗`
- **58% 是正例、42% 是负例**——所以读取器**必须按 `score` 过滤**，
  否则会给 42% 的样本贴上错误的"这是同义句"标签
- `max_length: 128`（问句都很短，实测平均每对 19.5 个字符）

### 2.2 CLIP ← Flickr30K

- 31,783 张图、每张 5 条 caption
- 发布方只给了 `test` split，train/validation 由 `split_modulo` 从中间切
- **同一个图有 5 条 caption**：分片按 **caption** 分桶（不是按行号），
  保证同一句话不会同时出现在训练和评测两边

### 2.3 Reranker ← MS MARCO

- 3.4 GB `corpus.jsonl` + `queries.jsonl` + `qrels/train.jsonl`
- MTEB 的排版是**三张表**，不是三元组，所以要先转换：

```bash
python tools/mteb_to_triplets.py \
  --dataset-dir /flagos-retrieval-datasets/msmarco \
  --out /flagos-retrieval-datasets/msmarco/triplets \
  --num-negatives 1 --max-queries 20000
```

- **负例是随机采的，不是挖出来的**。论文的配方是 hard negative mining，
  这里只做到随机负例——这是个已知弱点，不是数据本身的性质

---

## 三、代码改动：只加读取路径，不动训练逻辑

按 AGENTS.md §六（能改 FlagScale 就不动上游），改动全在 `FlagScale` 内，
且**没有改任何 loss 或训练循环**：

| 新增 | 位置 | 作用 |
|---|---|---|
| `SentencePairDataset` | `train/datasets/retrieval.py` | 句对 → `anchor`/`positive`，按 `score` 过滤 |
| `ClipParquetDataset` | 同上 | 从 parquet 读 `image`/`caption` |
| `data.format: pair \| parquet` | `build_retrieval_dataset` | 格式分派 |
| `tools/mteb_to_triplets.py` | 新增 | MS MARCO 三表 → 三元组 |
| `tools/modelscope_parallel_download.py` | 新增 | 并发下载（见 §五）|

**为什么加 `ClipParquetDataset` 而不是要求重新打包成 tar**：
`ClipTarDataset` 读 tar 是因为当初的 COCO dump 恰好是 tar 格式，
那是那份数据的偶然属性，不是训练器的要求。训练器只要 (图, 文) 一对。

### 3.1 两个容易静默出错的点（都有单测）

1. **不过滤 score**：LCQMC 有 42% 负例，全读进去等于教会模型把同义句推开。
   不报错、loss 照常下降、能力实际倒退。
2. **按行号分片**：Flickr30K 每图 5 条 caption，按行号分会把同一张图的
   caption 分到训练和评测两边。

单测：`tests/unit_tests/train/test_continued_pretraining_datasets.py`（9 项）

---

## 四、怎么跑

```bash
# 1. 下载（三台机器都装好了 modelscope）
bash examples/retrieval/download-continual-pretrain.sh lcqmc      # 或 flickr30k / msmarco / all

# 2. 训练（命令行只传七类参数，超参全在配置里）
python -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=bge_m3_continual_lcqmc \
  train.parallel=ddp experiment.runner.nproc_per_node=1 \
  train.hardware.device_type=cuda \
  train.output_dir=... experiment.exp_dir=...
```

三个配置：

| 配置 | 并行 | batch | epochs | lr |
|---|---|---|---|---|
| `bge_m3_continual_lcqmc` | ddp 1 卡 | 32 | 1 | 2e-5 |
| `clip_vit_large_flickr30k` | ddp 1 卡 | 64 | 1 | 1e-7 |
| `qwen3_reranker_msmarco` | fsdp2 2 卡 | 16 | 1 | 1e-5 |

> **lr 都沿用了各模型验收配置的值**，没有为新数据重新调。
> 这是刻意的：换数据的同时换 lr，就分不清效果来自哪一边。

---

## 五、下载慢：ModelScope 限速是**每连接**的

实测单连接只有 **130-155 kB/s**，4.4 GB 的 Flickr30K 单线程要 **7 小时**。
一开始以为是网络问题，测了 4 路并发才发现：

| 方式 | 吞吐 |
|---|---|
| 单连接 | 132 kB/s |
| **4 路并发** | **~580 kB/s（4.4×）** |
| 10 路并发（实际下载）| **~3.8 MB/s（28×）** |

**所以不要用 `snapshot_download` 拉大仓**——它一个文件一个文件下。
用 `tools/modelscope_parallel_download.py`：

```bash
python tools/modelscope_parallel_download.py lmms-lab/Flickr30K \
  --dest /flagos-retrieval-datasets/flickr30k --workers 10
```

两个实现细节：
- `-C -` 断点续传：限速下中途断掉不能从头来
- **按文件大小校验完成**：ModelScope 对不存在的路径返回 HTTP 200 + JSON 错误体，
  仅看状态码会把错误页当成下载成功

---

## 六、还没做的

| 项 | 说明 |
|---|---|
| **三个模型的训练都还没跑** | 本次只到"数据可下载 + 读取器可用 + 单测通过" |
| hard negative mining | MS MARCO 用的是随机负例，与论文配方有差距 |
| BGE 的多语言/多语料配方 | 论文预训练用了 1.2B 条 194 语料，这里只有 LCQMC 的 24 万条中文问句 |
| 续训的学习率曲线 | 只跑 1 epoch，没有做 lr schedule 的 re-anchor 验证 |

---

## 七、2026-11-12 实际启动记录

三平台同时启动，**两台跑通、一台机器不可用**。

| 平台 | BGE / LCQMC | CLIP / Flickr30K | 状态 |
|---|---|---|---|
| **A100** | ✅ 跑完 | ✅ 跑完 | 正常 |
| **PPU** | ✅ 跑完 | ⏳ 训练中 | 正常 |
| **昆仑芯** | ❌ 起不来 | ❌ | **机器不可用** |

**已完成的数值**（1 epoch，全链路含 test）：

| 平台 | 模型 | train_loss | validation_loss | test_loss |
|---|---|---|---|---|
| A100 | BGE-M3 / LCQMC | 0.007598 | 0.010034 | 2.1807 |
| PPU | BGE-M3 / LCQMC | 0.007648 | 0.009699 | 2.2018 |
| A100 | CLIP / Flickr30K | 0.197362 | 0.136002 | 0.1496 |

> ⚠️ **`test_loss` 不是检索指标，不能用来判断"继续预训练有没有用"。**
> 它是在 T²-Ranking 三元组上算的同一个对比/对称 loss，跨了语料和任务，
> 绝对值没有意义（BGE 的 2.18 比 train 的 0.0076 大 300 倍，是因为
> T²-Ranking 的负例更难、批次构成不同，不是模型变差了）。
>
> **要回答"有没有用"，必须用 [14](../results/14-inference-throughput.md) 那套
> rank-based 指标（R@1/5/10、MRR），base 与 trained 同一次调用测。**
> 这正是 AGENTS.md §九那条"训练完必须评测，不许只交 loss"的意思。

### 7.1 昆仑芯：`torch.mm` 本身挂死

**不是我们代码的问题。** 最小复现：

```python
import torch
x = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16)
y = torch.mm(x, x); torch.cuda.synchronize()   # 永不返回
```

- `torch.cuda.device_count()` 返回 8，`xpu-smi` 也正常显示
- 但**任何 kernel 都不执行**，连 `timeout 40` 都杀不掉（进程处于 R 状态满转）
- 训练侧的表现为：日志停在 `reproducibility: ...` 之后，**8 张卡各占 0.7 GiB 但模型从未上卡**

定位过程（每一步都排除了一个假设）：

| 假设 | 验证 | 结论 |
|---|---|---|
| 数据读不了 | 在昆仑芯上读 LCQMC 200 行 | ❌ 0.11 s 正常 |
| tokenizer 慢 | 单独加载 tokenizer | ❌ 0.9 s 正常 |
| `from_pretrained` 慢 | 单独加载 BGE 模型 | ❌ 0.9 s 正常 |
| **`model.to("cuda")` 慢** | 单独执行 | ✅ **卡死** |
| 是某张坏卡 | 逐卡 `torch.mm` | ✅ **全部挂死** |

> **按 AGENTS.md §二，跑任何东西之前先探卡。** 这次我直接启动了训练，
> 15 分钟才发现是机器问题。探卡脚本 10 行、<1 s，应该先跑。

### 7.2 两个真实 bug（都是这次跑出来的）

**(a) `format` 单一导致评测集读成空**

继续预训练配置里 `data.format: pair`（LCQMC），但 `test_path` 指向
T²-Ranking **三元组** parquet。读取器去找 `sentence1` 列，找不到 →
**静默产出 0 行** → 一个完整 epoch 训练完之后才报
`Evaluation dataset is empty for this split`。

修法：`{train,validation,test}_format` 三个 split 各自声明格式；
同时在 `build_split_loaders` 里**先探一个 batch**，不匹配就立刻报错。

> 这个 bug 的代价是 **A100 + PPU 各浪费一个完整 epoch**（各约 30 分钟）。
> 如果 loader 构建时就能报错，代价是 0 秒。

**(b) Flickr30K 的 caption 是 `list<string>`，不是字符串**

见 §3.1。`str(caption)` 得到 `"['a', 'b']"`，tokenizer 学的是方括号和引号。

### 7.3 顺带确认：BGE 继续预训练是有效的

两个平台独立跑出几乎相同的数：

| | train_loss | validation_loss |
|---|---|---|
| A100 | 0.007598 | 0.010034 |
| PPU | 0.007689 | 0.009904 |

**两个平台的一致性说明这条链路（数据读取 → 对比学习 → 存盘）是通的**，
且平台间数值差异远小于模型间差异。这本身就是一个可用的跨平台对齐证据。

### 7.4 BGE 继续预训练的实测效果：**变差了**

跑完 A100 上的 LCQMC 继续预训练后，把 `checkpoint-epoch-1` 拿去做
**rank-based 评测**（T²-Ranking test，2000 query × 16 候选，base 与 trained 同参数）。

**主表只报论文口径的指标**（BGE-M3 论文主指标是 **nDCG@10**）：

| 指标 | base | trained | 变化 |
|---|---|---|---|
| **nDCG@10** | **0.8602** | **0.8448** | **−1.54 点** |

<details>
<summary>完整指标（<b>非论文口径</b>，仅备查：BGE-M3 不报 recall / MRR）</summary>

| 指标 | base | trained | 变化 |
|---|---|---|---|
| R@1 | 0.7060 | 0.6730 | −3.30 点 |
| R@5 | 0.9635 | 0.9645 | +0.10 点 |
| R@10 | 1.0000 | 0.9995 | −0.05 点 |
| MRR | 0.8143 | 0.7939 | −2.04 点 |

</details>

**同一个 checkpoint 的 train_loss 是 0.0076、val_loss 0.0100——看起来"学得很好"。
但 nDCG@10 掉了 1.54 点**（R@1 与 MRR 同向，见折叠区）。

> **这是"loss 好 ≠ 模型好"最干净的一个反例。**
>
> ~~最初的解释是"同义句训练让模型把字面相近的句子拉近，与 T²-Ranking 的目标不同向"。~~
> **这个解释已被 §7.5 的逐对诊断推翻**：训练后 margin 和配对准确率都变好了，
> 崩的是相似度的绝对尺度。原因见 §7.5。

**在查清原因（见 §7.5）之前，不要把这批 checkpoint 当"改进版"用。** 可能的方向：

| 方向 | 依据 |
|---|---|
| 温度/尺度校准 | §7.5 显示正负例绝对相似度一起塌缩 |
| lr 调小或只训部分层 | 1 epoch、lr 2e-5 全参 |
| 换检索式语料（query-passage）| LCQMC 正例是改写句，不是相关段落 |
| 加 hard negative | 现在只有 in-batch 负例 |
| 混合原始任务数据 | 灾难性遗忘的常规缓解手段 |

### 7.5 追加诊断：BGE 变差的原因**不是**表征退化

上面 7.4 我给的解释是"同义句训练让模型只认字面相近"。**做了逐对诊断后，这个解释是错的。**

同一批 600 条 T²-Ranking test 行，两个模型各打分，看正负例的相似度分布：

| 模型 | 正例相似度 | 负例相似度 | **margin** | 配对准确率 |
|---|---|---|---|---|
| base | 0.6963 | 0.6030 | **0.0934** | 85.0% |
| **trained** | **0.5490** | **0.4049** | **0.1440** | **86.0%** |

**训练后 margin 反而更大（+54%），配对准确率也更高。** 但全量 R@1 掉了 3.3 点。

**两个口径为什么会矛盾？**

- 配对准确率只问"正例分数比**它自己的**负例高吗"——这是一个**局部**比较。
- R@1 要求正例在**该 query 的 16 个候选里**排第一——这是**全局**比较，
  而且候选来自不同 query 的负例池，绝对分数会互相竞争。

训练后**正例和负例的绝对相似度一起塌了**（0.696→0.549、0.603→0.405），
而塌缩幅度不同（neg 掉得更狠）。margin 变大是"两者都往 0 掉、负例掉得更快"
的结果。当候选池扩大、负例来自别的分布时，这种尺度塌缩会让排序更容易被
无关的高分候选干扰。

> **所以真正的失效模式是"相似度尺度塌缩"，不是"表征方向错误"。**
> 这也解释了为什么 loss 看起来很好（margin 确实变大了）、配对准确率也高
> （86% > 85%），但检索指标反而下降——**三个指标测的是三种不同的东西**。

**下一步该验的**（都未做）：

| 假设 | 怎么验 |
|---|---|
| 尺度塌缩导致全局排序劣化 | 对 trained 的 embedding 做温度/尺度校准后再算 R@1 |
| lr 过大 | 用 1e-6 重跑，看尺度塌缩是否缓解 |
| 需要混合原始检索任务数据 | 在 LCQMC 里掺入 T²-Ranking 式三元组 |

> **这条比 7.4 的结论更重要**：同类实验里如果只看 margin 或配对准确率，
> 会得出"继续预训练有效"的相反结论。**必须报 R@k 这类全池排序指标。**

### 7.6 Reranker 继续预训练（MS MARCO）：**也变差了，而且更严重**

A100、2 卡 FSDP2、MS MARCO 20,000 query 的三元组、1 epoch。
评测在同一份 T²-Ranking test 上跑 1000 query（base 与 trained 同参数）。

**主表只报论文口径的指标**（Qwen3-Reranker 报 **nDCG@10**）：

| 指标 | base | trained | 变化 |
|---|---|---|---|
| **nDCG@10** | **0.8819** | **0.8331** | **−4.88 点** |

<details>
<summary>完整指标（<b>非论文口径</b>，仅备查：论文不报 recall / MRR / mean_rank）</summary>

| 指标 | base | trained | 变化 |
|---|---|---|---|
| R@1 | 0.7350 | 0.6250 | −11.00 点 |
| R@5 | 0.9820 | 0.9770 | −0.50 点 |
| R@10 | 1.0000 | 1.0000 | 0 |
| MRR | 0.8423 | 0.7770 | −6.53 点 |
| mean_rank | 1.485 | 1.682 | +0.197 |

R@1 与 MRR 同样大跌，与 nDCG@10 同向。

</details>

训练侧：train_loss 0.009087、**val_loss 1.464651**（差 161 倍）。

> **val_loss 1.46 是明确的信号**：同一个 checkpoint 在训练分布上 loss 0.009，
> 换到 T²-Ranking 上立刻涨 161 倍。这不是"泛化差一点"，
> 是模型被 MS MARCO 的分布**拽走**了。

**为什么 R@1 掉 11 点（BGE 只掉 3.3 点）**：

| | BGE | Reranker |
|---|---|---|
| 训练数据 | LCQMC 同义句对（中文口语问句）| MS MARCO（英文 passage）|
| 负例 | in-batch | **随机从 884 万 doc 里采** |
| 与评测集的关系 | 都是中文短句，**领域接近** | **中↔英、短句↔段落，领域差得远** |

MS MARCO 的负例是随机采的，任务太容易（train_loss 0.009），模型学到的是
"区分完全不相关的英文段落"，而这个能力**对中文 reranking 是干扰**。

### 7.7 三个模型的一致结论

| 模型 | 数据 | R@1 变化 | MRR 变化 | 判定 |
|---|---|---|---|---|
| BGE-M3 | LCQMC（中文同义句）| **−3.30 点** | −2.04 点 | ❌ 变差 |
| Qwen3-Reranker | MS MARCO（英文段落）| **−11.00 点** | −6.53 点 | ❌ 变差 |
| CLIP | Flickr30K | −0.04 点 | −0.00 点 | ⚪ 持平（lr 太小，见 §7.8）|

**两个模型变差、一个持平、零个变好。** 但变差的两个 lr 是 1e-5～2e-5，
持平的那个 lr 是 1e-7——**"持平"更可能是"没训动"，不能算作正面证据**。

#### 结论要怎么写

**不能写"继续预训练失败了"，要写清前提。** 这三个 run 的共同特征：

1. **只有 1 epoch**，但 lr 沿用验收配置（BGE 2e-5、reranker 1e-5），
   没有为"继续预训练"这个更温和的目标调小
2. **数据领域与评测集不同**（中文问句 ↔ 中文检索、英文段落 ↔ 中文检索）
3. **没有混入任何原任务数据**——灾难性遗忘的标准触发条件
4. **负例质量低**：BGE 只有 in-batch，reranker 是随机负例

**所以更准确的说法是**：*在这套配置下（单 epoch、验收 lr、跨领域语料、弱负例），
继续预训练会让检索指标倒退。* 要判断"继续预训练本身有没有用"，
至少需要一次 lr 调小 + 混入原任务数据 + hard negative 的对照实验。

> **这三次实验的价值不在"成功"，而在给出了一条可复现的负结果**，
> 以及一个反直觉的发现（见 §7.5）：训练侧所有迹象（loss 低、margin 变大、
> 配对准确率升高）都是好的，只有全池排序指标是坏的。

### 7.8 CLIP 继续预训练（Flickr30K）：**基本持平**

A100、1 卡、Flickr30K 1 epoch，评测在 MSCOCO 5k（跨数据集）上：

| 方向 | 指标 | base | trained | 变化 |
|---|---|---|---|---|
| **text→image** | R@1 | 0.5446 | 0.5442 | −0.04 点 |
| | R@5 | 0.7794 | 0.7808 | +0.14 点 |
| | R@10 | 0.8538 | 0.8542 | +0.04 点 |
| | MRR | 0.6527 | 0.6527 | −0.00 |
| **image→text** | R@1 | 0.6216 | 0.6222 | +0.06 点 |
| | R@5 | 0.8464 | 0.8462 | −0.02 点 |
| | R@10 | 0.9086 | 0.9084 | −0.02 点 |
| | MRR | 0.7217 | 0.7220 | +0.03 点 |

**所有指标变化都 < 0.15 点，在噪声范围内。** 三个模型里只有 CLIP 没有变差。

**为什么 CLIP 没变差**：lr 是 **1e-7**（BGE 2e-5、reranker 1e-5），
低了 2-3 个数量级——这个量级下几乎不动原有权重，等于没训。

> **所以这一条不能读成"CLIP 适合继续预训练"，只能读成"lr 1e-7 太小、没起作用"。**
> 要验证 CLIP 的继续预训练是否有益，需要一个更高的 lr 做对照。

### 7.9 ⚠️ 一个数据划分的真实缺陷（已修）

**这个问题是甲方提问时查出来的，不是我自己发现的。**

`SentencePairDataset._files()` 原本 glob `data/*.parquet`，会匹配到数据集
**自带的全部 split 文件**，然后再叠加我们自己的 hash 分桶：

```
LCQMC 三个官方文件：train 238,766 / validation 8,802 / test 12,500 行
我们的 train split 产出 98,420 个不同 anchor
其中 3,046 个来自 LCQMC 的**官方 test 文件**          ← 泄漏
```

**对本目录所有结论的影响：零。** 因为三次实验的评测集全部是**另一个数据集**：

| 训练 | 评测 | 是否重叠 |
|---|---|---|
| LCQMC | T²-Ranking | ❌ 不同数据集 |
| MS MARCO | T²-Ranking | ❌ 不同数据集 |
| Flickr30K | MSCOCO 5k test | ❌ 不同数据集 |

**但这是一个随时会咬人的隐患**：只要有人在训练语料上评测（这条 pipeline 的常规用法），
就会静默泄漏。已修：`_filter_by_split()` 按文件名前缀过滤；
单文件 dump（T2Ranking、我们的 MS MARCO 三元组）没有 split 前缀，行为不变。

回归测试 `test_split_named_shards_are_not_mixed` 固定了这一点（含 `split="all"` 必须
读到全部文件的反向断言）。

> 附带影响：之前报的 **val_loss 不是干净的留出集**（验证集混入了官方 train 文件的行）。
> 三个模型的 val_loss 数值因此偏乐观，但结论方向不受影响——
> rank 指标是在完全独立的数据集上测的。
