# 04 · CLIP ViT-L/14

原文：`Papers/CLIP/clip_paper.tex`，ICML 2021，§2.2、附录 A.1/E.1、表 13。
模型权重：`openai/clip-vit-large-patch14`。

## 训练数据

### 预训练：WIT（WebImageText）

| 项 | 内容 |
|---|---|
| 规模 | **4 亿** (image, text) 对 |
| 构造 | 用 50 万条 query 从公开互联网搜索图文对；query 表 = 英文 Wikipedia 中出现 ≥100 次的词 + 高 PMI 二元组 + 高搜索量词条名 + WordNet synset |
| 平衡 | 每个 query 最多取 20,000 对，近似类平衡 |
| 总量 | 词数与 GPT-2 训练用的 WebText 相当 |
| 是否公开 | **未公开** |

### 微调：我们的做法（论文没有对应阶段）

**论文不做检索任务微调** —— CLIP 只做 WIT 预训练 + 零样本迁移。我们做的是
在全参微调，是论文里不存在的一步，所以**这一阶段的产物不能与论文对比**。

| 项 | 值 |
|---|---|
| 数据集 | **MSCOCO Captions 2017**（`wds_mscoco_captions2017`，webdataset tar 格式） |
| train | 21 个 shard，**19 GB** |
| test | 6 个 shard，含 `coco_test_karpathy.json`（**5000 images / 25,010 annotations**） |
| 微调方式 | **全参**微调（`freeze_backbone: false`），3 epoch |
| loss | 对称 InfoNCE（batch 内图文互负） |
| 超参 | `batch_size: 128`、`lr: 1e-7 → 1e-8` cosine、`max_length: 77` |

> **为什么用 MSCOCO 而不是 WIT**：WIT 未公开（见上）。MSCOCO 是 CLIP 论文的
> **评测**集之一，用它的 train split 微调、test split 评测，逻辑上与论文的
> "zero-shot 迁移"不同 —— 论文从没见过 COCO，我们训练时见过。
>
> ⚠️ 因此我们有"训练-测试同分布"的优势，**微调后的 R@1 高于论文的 zero-shot
> 数值是预期内的，不代表模型更强**。

### 训练数据可得性汇总

| 阶段 | 数据 | 可得性 |
|---|---|---|
| 论文预训练 | WIT 4 亿对 | ❌ 未公开 |
| 我们的微调 | MSCOCO Captions 2017 train（21 shard / 19 GB） | ✅ 公开 |

## 检索评测集与指标

论文只在这两个数据集上评**检索**（表 13）：

| 数据集 | 规模 | 指标 |
|---|---|---|
| **Flickr30k** | **1k test images** | R@1 / R@5 / R@10（双向） |
| **MSCOCO** | **5k test set** | R@1 / R@5 / R@10（双向） |

双向 = text→image 与 image→text 各报一遍。论文表 13 的列就是
`Flickr30k{R@1,R@5,R@10} MSCOCO{R@1,R@5,R@10}` × 两个方向。

> 原发布页：Flickr30k `shannon.cs.illinois.edu/DenotationGraph/`；
> MSCOCO `cocodataset.org/#download`。

## 论文用到的全部数据集（附录 A.1，27 个）

除上面两个检索集，CLIP 还在这 **27 个分类数据集**上做线性探针 / 零样本评测。
这张表是论文附录 Table A1 的原文，**完整无省略**：

| # | 数据集 | 类别数 | Train | Test | 评测指标 |
|---|---|---|---|---|---|
| 1 | Food-101 | 102 | 75,750 | 25,250 | accuracy |
| 2 | CIFAR-10 | 10 | 50,000 | 10,000 | accuracy |
| 3 | CIFAR-100 | 100 | 50,000 | 10,000 | accuracy |
| 4 | Birdsnap | 500 | 42,283 | 2,149 | accuracy |
| 5 | SUN397 | 397 | 19,850 | 19,850 | accuracy |
| 6 | Stanford Cars | 196 | 8,144 | 8,041 | accuracy |
| 7 | FGVC Aircraft | 100 | 6,667 | 3,333 | mean per class |
| 8 | Pascal VOC 2007 Classification | 20 | 5,011 | 4,952 | 11-point mAP |
| 9 | Describable Textures | 47 | 3,760 | 1,880 | accuracy |
| 10 | Oxford-IIIT Pets | 37 | 3,680 | 3,669 | mean per class |
| 11 | Caltech-101 | 102 | 3,060 | 6,085 | mean-per-class |
| 12 | Oxford Flowers 102 | 102 | 2,040 | 6,149 | mean per class |
| 13 | MNIST | 10 | 60,000 | 10,000 | accuracy |
| 14 | Facial Emotion Recognition 2013 | 8 | 32,140 | 3,574 | accuracy |
| 15 | STL-10 | 10 | 1,000 | 8,000 | accuracy（10 split 平均） |
| 16 | EuroSAT | 10 | 10,000 | 5,000 | accuracy |
| 17 | RESISC45 | 45 | 3,150 | 25,200 | accuracy |
| 18 | GTSRB | 43 | 26,640 | 12,630 | accuracy |
| 19 | KITTI | 4 | 6,770 | 711 | accuracy |
| 20 | Country211 | 211 | 43,200 | 21,100 | accuracy |
| 21 | PatchCamelyon | 2 | 294,912 | 32,768 | accuracy |
| 22 | UCF101 | 101 | 9,537 | 1,794 | accuracy（3 split 平均） |
| 23 | Kinetics700 | 700 | 494,801 | 31,669 | mean(top1, top5) |
| 24 | CLEVR Counts | 8 | 2,000 | 500 | accuracy |
| 25 | Hateful Memes | 2 | 8,500 | 500 | ROC AUC |
| 26 | Rendered SST2 | 2 | 7,792 | 1,821 | accuracy |
| 27 | ImageNet | 1,000 | 1,281,167 | 50,000 | accuracy |

其中 12 个来自 Kornblith 等人的标准评测套件，15 个是论文新增的。
**Country211 和 Rendered SST2 是 CLIP 论文自己构造的**：

- Country211：从 YFCC100M 过滤出 211 个 ISO-3166 国家（每国 ≥300 张带 GPS 的照片），
  每国取 200 train + 100 test
- Rendered SST2：把 SST-2 的句子渲染成 448×448 黑白文字图，测 OCR 能力

> 需要完整清单的理由：论文报的是这 27 个的**平均**，只看其中几个无法复现它的
> 汇总数字。

### 训练数据 vs 评测数据的分工

| 用途 | 数据集 | 规模 | 论文是否公开 |
|---|---|---|---|
| 预训练 | **WIT** | 4 亿对 | ❌ |
| **检索评测** | **Flickr30k** | 1k test images | ✅ |
| **检索评测** | **MSCOCO** | 5k test set | ✅ |
| 迁移评测 | 27 个分类数据集 | 见上表 | ✅ 多数公开 |
| **我们的微调** | MSCOCO Captions 2017 **train** | 21 shard / 19 GB | ✅ |

> **Flickr30k 我们没有**：数据盘上只有 `t2ranking` 和 `wds_mscoco_captions2017`，
> 没有 Flickr 目录，代码里也没有引用。所以现在只测 MSCOCO 一个检索集，
> 论文是 Flickr30k + MSCOCO 两个都报。
>
> 要对齐论文需要补 Flickr30k（原发布页 `shannon.cs.illinois.edu/DenotationGraph/`，
> 1k test images，双向 R@1/5/10）。这是**评测侧**缺口，不影响训练。

## 一个容易漏的细节

> 论文原文：*"For both these datasets we prepend the prompt `a photo of` to the
> description of each image which we found boosts CLIP's zero-shot R@1
> performance between 1 and 2 points."*

**评测时文本侧要加 `a photo of` 前缀**，能涨 1-2 点 R@1。我们目前的评测
没加这个前缀，是个可修正项。

## 对齐方案

**已与论文一致的部分**：我们用 MSCOCO Captions 2017 的 `test` 目录做全库检索
（N=5000，双向 5000×5000），与论文的 "5k test set" 口径一致。

**已确认（2026-10-29 实测）**：`coco_test_karpathy.json` 里有 **5000 images /
25,010 annotations**，正是官方 Karpathy test split，不是随机子集。

**仍需修正**：

1. **文本侧加 `a photo of` 前缀** —— 论文实测能涨 1-2 点 R@1，我们目前没加
2. 论文报的是 **zero-shot**；我们是微调后且微调用的是 COCO train，
   **不可直接比**，对比须注明
