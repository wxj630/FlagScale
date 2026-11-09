# 16 · 华为昇腾 910C 结果

**状态（2026-11-05）：BGE-M3 / CLIP 已完成；Reranker 正在跑 1 epoch（4 卡 FSDP2）。**

- 硬件：Ascend 910C（`Ascend910_9382`），64 GB HBM/芯片
- 软件：CANN 8.5.0，torch 2.6.0+cpu，torch_npu 2.6.0.post5，transformers 4.57.6
- 运行时：原生 `npu` 设备 + HCCL，**不经过 Torch-FL**
- 数据：T²-Ranking test（16,740 query，1 正 + 15 负）/ MSCOCO 5k（双向 5,000×5,000）
- 训练配方：`master_dtype=fp32` + bf16 autocast；BGE 单卡 + 梯度检查点；Reranker 4 卡 FSDP2

> ⚠️ 本文所有吞吐都是**未优化基线**：`train.hardware.use_flaggems=false`，
> 没有挂 FlagGems kernel。昇腾上尚未验证 FlagGems。

> 💡 训练吞吐（含反向+优化器）与推理吞吐是两个口径，不要混比。
> 纯前向的推理吞吐需另测。

## 零、A100 对照总表（验收口径）

| 项 | A100 | 昇腾 910C | 相对 A100 | 达标线 | 判定 |
|---|---|---|---|---|---|
| **精度** BGE base nDCG@10 | 0.8559 | **0.8546** | **99.85%** | ≥95% | ✅ |
| **精度** BGE trained nDCG@10 | 0.8337 | **0.8313** | **99.7%** | ≥95% | ✅ |
| **精度** CLIP base t→i R@1 | 0.5458 | **0.5430** | **99.5%** | ≥95% | ✅ |
| **精度** CLIP base i→t R@1 | 0.6232 | **0.6216** | **99.7%** | ≥95% | ✅ |
| **精度** CLIP trained t→i R@1 | 0.7400 | **0.7390** | **99.9%** | ≥95% | ✅ |
| **精度** CLIP trained i→t R@1 | 0.7784 | **0.7778** | **99.9%** | ≥95% | ✅ |
| **精度** Reranker | — | 1 epoch 进行中 | — | ≥95% | 待定 |
| **吞吐** BGE（样本/s） | **118.0** | **55.25** | **46.8%** | ≥80%×系数 | ❌ 55.8%（门槛 99.1），需 1.79× |
| **吞吐** CLIP（样本/s） | **119.0** | **95.02** | **79.9%** | ≥80%×系数 | ❌ **95.1%**（门槛 100.0），需 1.05× |
| **吞吐** Reranker（对/s） | 7.62（2 卡） | 1 epoch 进行中 | — | — | 待定 |

> 系数：910C 实测 bf16 GEMM 316.02 TFLOPS（A100 的 119%）、带宽 1259.7 GB/s
> （A100 的 79%），不含 FP32 的三项几何平均 = **1.050**
> （见 [10-microbench-discount.md](../analysis/10-microbench-discount.md)）。
> ⚠️ **这个系数是按 docx §4.2 建议的算法我们自己算的，不是验收方给定的数值**——
> docx 只给公式和 `× 0.8`，系数部分原文标为"建议讨论"。
> 门槛 = A100 × 1.050 × 0.8：
> BGE 99.12 样本/s，CLIP 99.96 样本/s。

> ⚠️ **本节判定用的是我们实测的系数。** 验收方另有一套（其 BF16/FP32 倍数换算），
> 用它算的达成率更低。**两套都必须报**，对照表见
> [14-inference-throughput.md](14-inference-throughput.md) §二·五 与
> [10-microbench-discount.md](../analysis/10-microbench-discount.md) §一·六。


**一句话**：CLIP 精度与 A100 几乎一致，吞吐只差 5%；BGE 精度也一致（trained nDCG
99.7%），但吞吐只有 A100 的 46.8%，主要不是 GEMM——昇腾的 bf16 GEMM 比 A100 快，
差距在 **RMSNorm / LayerNorm / GELU / softmax / 逐元素 / 小算子**。

## 一、BGE-M3（568M，1 卡，3 epoch）

| 项 | 值 | vs A100 |
|---|---|---|
| 训练耗时 | **83,283.8 s = 23.13 h** | **2.14× 慢** |
| 训练文本数 | 4,602,000 | — |
| **吞吐（端到端）** | **55.25 样本/s** | **46.8%** |
| **步时间** | **约 3.62 s/step**（200 texts/step） | **2.25× A100 的 1.605 s** |
| test_loss | 2.8727 | — |
| 并行 | DDP，1 卡 | — |
| 显存 | batch 10 × GA 4，梯度检查点，约 18 GB | — |

> 23,010 optimizer steps（4,602,000 / 200）。GEMM 微基准比 A100 快，
> 但整步慢 2.25×，说明差距主要来自 **非 GEMM 算子**，不是矩阵乘。

**检索指标（base / trained）**

| | R@1 | R@5 | R@10 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| 昇腾 base | 0.6959 | 0.9626 | 1.0000 | 0.8069 | **0.8546** |
| A100 base | 0.6992 | 0.9629 | 1.0000 | 0.8086 | **0.8559** |
| 相对 | 99.5% | 99.97% | 100.0% | 99.8% | **99.85%** |
| 昇腾 trained | 0.6454 | 0.9700 | 0.9996 | 0.7759 | **0.8313** |
| A100 trained | 0.6508 | 0.9685 | 0.9997 | 0.7791 | **0.8337** |
| 相对 | 99.2% | 100.2% | 100.0% | 99.6% | **99.7%** |

> BGE base 评测已完成；base nDCG@10 是 A100 的 99.85%。
> 全量 base eval 的纯前向吞吐为 160.3 样本/s（267,840 次编码 / 1,670.8s）；
> 本文第六节用与 14 文档一致的 2,000-query 协议，故是 155.2 样本/s。

## 二、CLIP ViT-L/14（428M，1 卡，3 epoch）

| 项 | 值 | vs A100 |
|---|---|---|
| 训练耗时 | **3,354.2 s = 0.932 h** | **1.25× 慢** |
| 训练样本 | 318,720 pairs | — |
| **吞吐（端到端）** | **95.02 样本/s** | **79.9%** |
| test_loss | 0.07135 | — |

### text → image

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| base | 0.5430 | 0.7794 | 0.8546 |
| trained | **0.7390** | **0.9154** | **0.9570** |
| Δ | **+19.60 pt** | +13.60 | +10.24 |

### image → text

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| base | 0.6216 | 0.8462 | 0.9102 |
| trained | **0.7778** | **0.9402** | **0.9710** |
| Δ | **+15.62 pt** | +9.40 | +6.08 |

与 A100 对比：

| 指标 | A100 | 昇腾 | 相对 |
|---|---|---|---|
| base t→i R@1 | 0.5458 | 0.5430 | 99.5% |
| base i→t R@1 | 0.6232 | 0.6216 | 99.7% |
| trained t→i R@1 | 0.7400 | 0.7390 | 99.9% |
| trained i→t R@1 | 0.7784 | 0.7778 | 99.9% |
| **吞吐** | **119.0** | **95.02** | **79.9%** |

CLIP 的微调增益（+19.6 / +15.6 点）与 A100、昆仑芯、PPU 完全同向，
说明训练本身正常；吞吐距门槛只差 **1.05×**。

## 三、Qwen3-Reranker-8B（4 卡 FSDP2，1 epoch 进行中）

- 并行：FSDP2，4 卡，batch 16/rank，GA 1（等效 batch 64 行/步 = 128 对/步）
- 精度：fp32 master + bf16 autocast；梯度检查点开启
- 卡：chip 2,3,14,15
- 当前：epoch 1，step 30，loss 0.4057
- 预计 1 epoch：约 18 h（4,783 step，按约 4.3 step/min）

**已经踩过并修掉的两个问题：**

1. **4 卡第一次 all-gather OOM**：每个 rank 先把完整 fp32 8B 放进 HBM（约 33 GB）。
   修复：FSDP2 先在 **CPU** 上建模型并分片，再把本 rank 的 shard 搬到 NPU
   （`730acdd3`，`NPU memory after FSDP2 move: allocated=7.6 GiB`）。
2. **epoch 结束 HCCL all-reduce `at::kDouble`**：`reduce_metrics` 用 float64，
   HCCL 不支持。修复：NPU 上用 float32（`ascend-hccl-float64-20261105.patch`）。

> 2 卡不可行：优化器状态约 131 GB，2×64 GB 连状态都放不下；
> 4 卡是当前配方的最低配置。本文不把 4 卡吞吐与 A100 2 卡直接比，
> 等 1 epoch 跑完再按“对/s”和每卡效率单独报告。

## 四、关键慢算子分析（重点）

### 4.1 方法

用 `scripts/profile_op_micro_npu.py` 在空闲 chip 上测单算子，形状与
A100 的 `docs/retrieval/07-a100.md` 微基准完全相同：

- bf16，B=8，S=512，H=1024，FFN=4096，vocab=151669
- 每项 warmup 10 次、30 次取中位数
- 原始 JSON：`docs/retrieval/data/op-micro/ascend_910c.json`

> 这是**微基准**，不是整步 profiler；它用来找候选慢算子，不能直接按倍数排优先级。
> 真实 step 的权重仍需用 `profile_step_segments.py` 在 NPU 上补测。

### 4.2 逐算子结果（ms/次）

| 算子 | 昇腾 910C | A100 | 昇腾/A100 |
|---|---|---|---|
| `mm_lm_head` (4096×1024 @ 1024×151669) | **7.901** | 15.62 | **0.51×（更快）** |
| `mm_ffn` (4096×1024 @ 1024×8192) | **0.266** | 0.352 | **0.76×（更快）** |
| `mm_hidden` (4096×1024 @ 1024×1024) | **0.076** | 0.092 | **0.83×（更快）** |
| `flash_attn_fwd` (B8 S512 GQA) | 1.819 | 1.440 | 1.26× |
| `rms_norm` | **0.156** | 0.024 | **6.50×** |
| `layer_norm` | **0.095** | 0.034 | **2.79×** |
| `silu` | 0.047 | 0.021 | 2.22× |
| `gelu` | 0.053 | 0.021 | 2.52× |
| `add` | 0.045 | 0.017 | 2.67× |
| `mul` | 0.044 | 0.017 | 2.59× |
| `sum_last` | 0.055 | 0.019 | 2.91× |
| `softmax` | 0.073 | 0.026 | 2.81× |
| `bmm_scores` (B8 128×128) | 0.098 | 0.033 | 2.98× |
| `cross_entropy` | 0.275 | 0.085 | 3.23× |

### 4.3 结论：昇腾是“非 GEMM 受限”，和 PPU 相反

- **GEMM 全部比 A100 快**：大 LM head 0.51×、FFN 0.76×、hidden 0.83×。
  这与 bf16 GEMM 微基准 316 TFLOPS（A100 的 119%）一致。
- **attention 只慢 1.26×**，不是主要矛盾。
- **RMSNorm 慢 6.5×**，是单项最差；LayerNorm 2.8×。
- **GELU / SiLU / add / mul / sum / softmax / bmm 慢 2.2–3.2×**。
- BGE 的 FFN 用 GELU + LayerNorm，Reranker 用 RMSNorm + SiLU，
  所以 BGE/Reranker 的慢点都在**归一化、激活、逐元素和小矩阵**，
  不在 GEMM。

  > ⚠️ **口径提醒**：上面 PPU 的「GEMM 占差距 44%」和昆仑芯的「GELU 占 66%」
  > 是**整步 profiler 的权重**（占与 A100 差距的百分比）；本节昇腾的
  > 「RMSNorm 6.5×」是**微基准倍数**，两者不可直接对比。
  > 昇腾的真实 step 权重还没测（见 4.1 的说明）。

**按模型看：**

| 模型 | 主要慢点 | 说明 |
|---|---|---|
| BGE-M3 | **LayerNorm / GELU / softmax / 逐元素** | GEMM 反而快；BGE 吞吐 46.8% 的差距不能用 GEMM 解释 |
| CLIP | **LayerNorm / quick_gelu / attention / 逐元素** | 吞吐 79.9%，距门槛仅 1.05×；conv/patch embed 需单独测 |
| Reranker | **RMSNorm 6.5× / SiLU 2.2× / softmax 2.8×** | 4 卡 FSDP2，1 epoch 中；GEMM 快，瓶颈在 norm/激活/通信 |

> 下一步应按“每步多花多少毫秒”排序，而不是按上表的倍数：
> 用 `profile_step_segments.py --device npu` 跑真实 BGE step，把
> `rms_norm/layer_norm/gelu/silu/softmax` 的调用次数乘进去。
> 目前可以确定的是：**不要先优化 GEMM**，它不是昇腾的短板。

### 4.4 FlagGems

昇腾上尚未安装/验证 FlagGems；所有数字都是 `use_flaggems=false` 的厂商基线。
在昆仑芯和 PPU 上 FlagGems 分别 9/10 和 12/13 个算子更慢，
因此昇腾也不应默认假设换 FlagGems 能提速；先用真实 step profiler 定位，
再让厂商在 RMSNorm/LayerNorm/激活/softmax 的原生 kernel 上优化。

## 五、吞吐判定与下一步

| 模型 | A100 | 门槛（×1.050×0.8） | 昇腾 | 达成门槛 | 需提速 |
|---|---|---|---|---|---|
| BGE-M3 | 118.0 | 99.12 | 55.25 | **55.8%** | **1.79×** |
| CLIP | 119.0 | 99.96 | 95.02 | **95.1%** | **1.05×** |
| Reranker | 7.62（2 卡） | — | 1 epoch 中 | — | 待定 |

**优先级：**

1. **CLIP**：只差 1.05×，先补真实 step profiler；LayerNorm/quick_gelu/attention
   的小优化或减少 cast 就可能补齐。
2. **BGE**：差距 1.79×，重点在 LayerNorm/GELU/softmax/逐元素，不在 GEMM；
   需要厂商原生 bf16 kernel 或融合。
3. **Reranker**：先把 1 epoch 跑完并评测；4 卡 FSDP2 的通信/分片开销也要计入，
   不能只看单算子。

## 六、推理吞吐（只计模型前向）

口径与 [14-inference-throughput.md](14-inference-throughput.md) 一致：

- 只计模型前向；数据加载、tokenize 后的 CPU 侧排序都在计时区间外
- 每次前向前后 `synchronize`
- BGE 2,000 query × 16 候选 = 32,000 次编码，batch 32
- CLIP 5,000 图文对 = 10,000 次编码，batch 64
- Reranker 300 query × 16 = 4,800 对，batch 4
- 都用 **base 权重**，`InferenceClock` 计时；原始 JSON 在
  `docs/retrieval/data/inference/ascend_*.json`

| 模型 | 计数量 | 模型耗时 | **推理吞吐** | vs 昆仑芯 | vs PPU |
|---|---|---|---|---|---|
| **BGE-M3** | 32,000 | 206.13 s | **155.24 样本/s** | **96.5%** | **209.5%** |
| **CLIP ViT-L/14** | 10,000 | 25.58 s | **390.99 样本/s** | **121.8%** | **144.3%** |
| **Qwen3-Reranker-8B（base）** | 4,800 | 208.28 s | **23.05 对/s** | **108.9%** | PPU 测不了 |

训练 vs 推理（同平台，粗略量级）：

| 模型 | 训练吞吐 | 推理吞吐 | 倍数 |
|---|---|---|---|
| BGE-M3 | 55.25 样本/s | **155.24** | **2.81×** |
| CLIP | 95.02 样本/s | **390.99** | **4.11×** |
| Reranker | 4 卡训练中 | 23.05 对/s（单卡） | 卡数不同，不直接比 |

> **这组数字很关键**：纯前向下昇腾 BGE 只比昆仑芯慢 3.5%，CLIP 反而快 22%，
> Reranker base 比昆仑芯快 9%。训练吞吐却明显落后，说明差距主要在
> **反向、优化器和非 GEMM 小算子**，不是前向 GEMM/推理能力。
> 这也和第四节的逐算子结果一致：GEMM 比 A100 快，慢的是 norm/激活/逐元素。

## 七、复现

```bash
# 单算子微基准（本文第四节）
ASCEND_RT_VISIBLE_DEVICES=4 \
  python scripts/profile_op_micro_npu.py \
  --out FlagScale/docs/retrieval/data/op-micro/ascend_910c.json

# 训练
bash examples/retrieval/ascend/run-ascend-bge-gc.sh
EPOCHS=1 bash scripts/run-ascend-reranker-4card-bs8.sh

# 评测（全量 test + predictions）
python examples/retrieval/evaluate_retrieval.py ... --save-predictions
python tools/recompute_metrics.py <predictions.json> --ndcg
```

## 八、限制

- Reranker 只跑 1 epoch，尚无最终精度。
- 推理吞吐用 base 权重；trained 权重的推理吞吐尚未单独测。
- 微基准未按真实 step 调用次数加权；不能把 6.5× 直接当成 65% 的吞吐损失。
- 未测 CLIP 的 conv/patch-embed 和 reranker 的 RoPE/通信开销。
- 未验证完整 3 epoch；A100/昆仑芯的 reranker 长跑都出现过发散，
  昇腾不应直接上 3 epoch。
