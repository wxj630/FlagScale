# 14 · 推理吞吐（三平台实测）

**这是"推理"口径，与 [07](07-a100.md)–[09](09-ppu.md) 的训练吞吐不是一回事。**
训练吞吐含反向传播与优化器，推理只含前向；两者相差 2-3 倍是正常的，
不该混在同一张表里比较。

## 一、口径（先看，否则数字没意义）

| 项 | 定义 |
|---|---|
| **测量范围** | **只计模型前向**。数据加载、tokenize 后的 CPU 侧排序全部在计时区间外 |
| **同步** | 每次前向前后各 `synchronize` 一次，否则测到的是 kernel 发射延迟而不是计算时间 |
| **单位** | embedding 任务 = **文本编码数/s**；reranker = **查询-文档对/s** |
| **batch** | BGE 32、CLIP 64、Reranker 4（reranker 受显存限制） |
| **样本量** | BGE 2,000 query × 16 候选 = 32,000 次编码；CLIP 5,000 图文对 = 10,000 次编码；Reranker 300 query × 16 = 4,800 对 |
| **实现** | `examples/retrieval/evaluate_retrieval.py` 的 `InferenceClock`（`1f7e9709` + `20a9242f`） |
| **原始产物** | `docs/retrieval/data/inference/*.json`（每份都含 `inference` 字段，可复核） |

> **为什么必须重新测，不能从旧结果反推**：历史 eval JSON 只有
> `model_type/task/checkpoint/split/samples/metrics`，**没有任何时间字段**；
> 文件 mtime 又混着模型加载和几百 MB 的 predictions 写盘，无法反推。

## 二、实测结果

### BGE-M3（568M）

| 平台 | 编码数 | 模型耗时 | **推理吞吐** | vs 昆仑芯 |
|---|---|---|---|---|
| **昆仑芯 P800** | 32,000 | 198.9 s | **160.9 样本/s** | 基准 |
| **PPU ZW810E** | 32,000 | 431.8 s | **74.1 样本/s** | **46.1%** |
| A100 | — | — | ⏳ 待测 | — |

### CLIP ViT-L/14（428M）

| 平台 | 编码数 | 模型耗时 | **推理吞吐** | vs 昆仑芯 |
|---|---|---|---|---|
| **昆仑芯 P800** | 10,000 | 31.1 s | **321.1 样本/s** | 基准 |
| **PPU ZW810E** | 10,000 | 36.9 s | **270.9 样本/s** | **84.4%** |
| A100 | — | — | ⏳ 待测 | — |

### Qwen3-Reranker-8B

| 平台 | 对数 | 模型耗时 | **推理吞吐** | 说明 |
|---|---|---|---|---|
| **昆仑芯 P800** | 4,800 | 226.8 s | **21.2 对/s** | 预训练权重（base） |
| PPU ZW810E | — | — | ❌ 测不了 | 8B 单卡放不下，需 FSDP2 分片；见 §四 |
| A100 | — | — | ⏳ 待测 | — |

## 三、训练 vs 推理（同平台、同模型）

| 模型 | 平台 | 训练吞吐 | 推理吞吐 | 倍数 |
|---|---|---|---|---|
| BGE-M3 | 昆仑芯 | 54.0 样本/s | **160.9** | **2.98×** |
| CLIP | 昆仑芯 | 83.4 样本/s | **321.1** | **3.85×** |
| Reranker | 昆仑芯 | 6.3 对/s | **21.2 对/s** | **3.37×** |
| BGE-M3 | PPU | 67.0 样本/s | **74.1** | 1.11× |
| CLIP | PPU | 69.1 样本/s | **270.9** | 3.92× |

> ⚠️ **上表混了口径，只能当粗略量级参考**：
> 训练吞吐是 run 全程折算（含评测、存盘、卡竞争），推理是纯前向；
> BGE 在 PPU 上只有 1.11× 特别可疑——训练那 67.0 是端到端值，
> 而推理的 74.1 是纯算力，说明 PPU 的 BGE 训练里非计算开销占比不小。

## 四、两个说明

**1. PPU 的 reranker 目前测不了。** 8B 模型单卡放不下
（fp32 master 约 33 GB + 激活），报 `CachingDeviceAllocator: invalid device index`。
要测得用 FSDP2 分片到两卡，而评测脚本 `evaluate_retrieval.py` 是单卡设计
（不需要 torchrun）。这是个已知缺口，不是数字缺失。

**2. 昆仑芯 reranker 用的是 base 权重。** 训练还在 epoch 1，没有 checkpoint。
等训完重测才有 trained 的推理吞吐。

## 五、复现命令

```bash
# 昆仑芯（卡 2，单卡）
python examples/retrieval/evaluate_retrieval.py \
  --model-type bge_m3 --task embedding \
  --model-path /flagos-retrieval-models/bge-m3/bge-m3 \
  --data-path /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15 \
  --split test --max-samples 2000 --num-negatives 15 --batch-size 32 \
  --device cuda --max-length 512 \
  --output infer_bge.json
```

```bash
# PPU（容器 flagos-retrieval 内，需 GEMS_VENDOR=thead 与 NCCL_SOCKET_IFNAME=bond0）
export FLAGSCALE_DEVICE=flagos FS_PLATFORM=flagos FLAGSCALE_DISTRIBUTED_BACKEND=flagos
export FLAGSCALE_AUTOCAST_DEVICE=flagos GEMS_VENDOR=thead NCCL_SOCKET_IFNAME=bond0
python examples/retrieval/evaluate_retrieval.py ... --device flagos ...
```

> `GEMS_VENDOR` 漏了会报
> `FlagGems vendor detection failed for FLAGOS_ACCELERATOR='ppu'`——
> `run-ppu-three.sh` 里有这一行，手写命令容易忘。
