# 09 · PPU ZW810E 结果

**满足 docx §4 吗？精度 ✅ 达标，吞吐 ❌ 不达标。**

- 硬件：PPU ZW810E（平头哥），96 GB/卡
- 软件：Torch-FL `flagos` 设备，torch 2.10.0，torch_fl 2.10.0+ppu2.0.0，transformers 4.57.0
- 评测：T²-Ranking test（16,740 query）/ MSCOCO 5k（双向 5000×5000）
- 日期：2026-10-30


> **数据来源说明**：本文档的检索指标由 `tools/recompute_metrics.py --ndcg`
> 从 `*.predictions.json` 重算，与当时 `metrics.json` 记录的数值可能有
> 0.001 级的差异（原评测按固定 cutoff 统计，重算用了完整候选与 nDCG 增益），
> **跨平台对比统一以本文档口径为准**。重算不需要重跑模型。


> ⚠️ **本文件所有结果均未引入 FlagGems。**
> `train.hardware.use_flaggems=false`（`conf/train.yaml` 与各平台启动脚本一致），
> 两个国产平台都走厂商 torch / Torch-FL 的默认算子，**没有挂任何 FlagGems kernel**。
> 也就是说：下面所有精度与吞吐数字都是**未优化**的基线，
> 算子级剖析给出的差距正是 FlagGems 要补的部分，尚未生效。

## 一、达标判定（对照 docx §4）

| 项 | 要求 | 实测 | 判定 |
|---|---|---|---|
| **精度**（BGE） | ≥ A100 的 95% | **99.9%** | ✅ **达标** |
| **精度**（CLIP t→i） | ≥ 95% | **100.0%** | ✅ **达标** |
| **精度**（CLIP i→t） | ≥ 95% | **100.0%** | ✅ **达标** |
| **吞吐**（BGE） | ≥ A100 × 0.8 | **56.8%**（系数=1）<br>**99.8%**（系数 0.711） | ⚠️ **差 0.18%**（差 0.12 样本/s） |
| **吞吐**（CLIP） | 同上 | **58.1%**（系数=1）<br>**102.1%**（系数 0.711） | ✅ **达标** |
| **吞吐**（Reranker，2 卡） | 同上 | **70.8%**（系数=1）<br>**124.4%**（系数 0.711） | ✅ **达标** |

> 系数取自 `10-microbench-discount.md`：PPU 810E 实测
> bf16 GEMM 136.43 TFLOPS（A100 的 **52%**）、带宽 2097.1 GB/s（**132%**），
> 不含 FP32 的三项几何平均 = **0.711**。
>
> **这个系数把结论翻转过来了**：按系数=1 时三项全不达标，
> 按芯片真实微基准算，CLIP 与 Reranker 都达标，BGE 只差 0.12 样本/s。

> 用真实系数判定后，只有 BGE 差一点点。它的瓶颈在 `bmm`（慢 15.3×）之外的
> 通用归约与归一化路径，属于 FlagGems 的覆盖范围。

## 二、BGE-M3（568M，1 卡，3 epoch）

| 项 | 值 | vs A100 |
|---|---|---|
| 训练耗时 | **69,132.4 s = 19.20 h** | **1.77× 慢** |
| 训练样本 | 4,602,000 texts | — |
| **吞吐（端到端）** | **67.0 样本/s** | **56.8%** |
| test_loss | 2.8759 | — |

**吞吐波动**：❌ **wandb 无记录**。

> PPU 的 bge 是 protobuf 损坏修复前启动的，wandb 初始化失败、整场没上云
> （见 `docs/retrieval-multiplatform.md` 的 wandb 根因）。日志也无每步耗时，
> 因此**只能给端到端 67.0，给不出最快/最慢/平均**。
> 要补：重跑时先验证 wandb 正常（`wandb.init` 成功），或加逐段时间戳。

| | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@1 | nDCG@5 | nDCG@10 | nDCG@20 |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.6988 | 0.9629 | 1.0000 | 1.0000 | 0.8084 | 0.6988 | 0.8426 | **0.8558** | 0.8558 |
| trained | 0.6548 | 0.9709 | 0.9996 | 1.0000 | 0.7824 | 0.6548 | 0.8261 | 0.8362 | 0.8363 |
| **Δ** | **−4.40 pt** | +0.80 | −0.04 | 0 | −2.60 | −4.40 | −1.65 | −1.96 | −1.95 |

### 与 A100 逐项对比

| 指标 | A100 | PPU | 相对 |
|---|---|---|---|
| base R@1 | 0.6992 | 0.6988 | 99.9% |
| base R@5 | 0.9629 | 0.9629 | **100.0%** |
| base MRR | 0.8086 | 0.8084 | 99.98% |
| base nDCG@10 | 0.8559 | 0.8558 | 99.99% |

## 三、CLIP ViT-L/14（428M，1 卡，3 epoch）

| 项 | 值 | vs A100 |
|---|---|---|
| 训练耗时 | **4,613.1 s = 1.281 h** | **1.72× 慢** |
| 训练样本 | 318,720 pairs | — |
| **吞吐（端到端）** | **69.1 样本/s** | **58.1%** |
| test_loss | 0.0701 | — |

**吞吐波动**：❌ **wandb 无记录**（同上，该容器 wandb 当时不可用）。

### text → image

| | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@1 | nDCG@5 | nDCG@10 | nDCG@20 |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.5458 | 0.7806 | 0.8560 | 0.9230 | 0.6536 | 0.5458 | 0.6728 | **0.6972** | 0.7142 |
| trained | 0.7390 | 0.9164 | 0.9564 | 0.9792 | 0.8174 | 0.7390 | 0.8365 | 0.8496 | 0.8554 |
| **Δ** | **+19.32 pt** | +13.58 | +10.04 | +5.62 | +16.38 | +19.32 | +16.37 | +15.24 | +14.12 |

### image → text

| | R@1 | R@5 | R@10 | R@20 | MRR | nDCG@1 | nDCG@5 | nDCG@10 | nDCG@20 |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.6234 | 0.8470 | 0.9106 | 0.9488 | 0.7229 | 0.6234 | 0.7445 | 0.7651 | 0.7748 |
| trained | 0.7786 | 0.9414 | 0.9712 | 0.9878 | 0.8497 | 0.7786 | 0.8686 | 0.8783 | 0.8826 |
| **Δ** | **+15.52 pt** | +9.44 | +6.06 | +3.90 | +12.68 | +15.52 | +12.41 | +11.32 | +10.78 |

### 与 A100 逐项对比

| 指标 | A100 | PPU | 相对 |
|---|---|---|---|
| base t→i R@1 | 0.5458 | 0.5458 | **100.0%** |
| base i→t R@1 | 0.6232 | 0.6234 | **100.0%** |
| base t→i nDCG@10 | 0.6967 | 0.6972 | 100.1% |
| trained t→i R@1 | 0.7400 | 0.7390 | 99.9% |

## 四、Qwen3-Reranker-8B（2 卡 FSDP2）

| 项 | 值 |
|---|---|
| 并行 | FSDP2，2 卡（ppu12/ppu13） |
| 显存 | 94.9 / 93.9 GB（96 GB 卡的 97%） |
| **吞吐（2 卡合计）** | **3.9 对/s** |
| 状态 | ⏳ **训练中**（已跑 8,324 步） |
| loss | 0.048（健康，单调下降） |

**吞吐波动**（wandb run `eoxqu2xl`，526 个稳态区间）：

| 统计 | 值 |
|---|---|
| 最慢区间 | **3.7 对/s** |
| 平均 | **3.9 对/s** |
| 最快区间 | **4.2 对/s** |
| 波动幅度 | **−5% ~ +8%** |

> 波动很小（±8%），说明 FSDP2 的步速稳定。
> 但吞吐只有昆仑芯 Megatron PP 的 **61%**（3.9 vs 6.4 对/s），
> 是 A100 FSDP2 的 **71%**（3.9 vs 5.51 对/s）——同样 2 卡 FSDP2，
> PPU 比 A100 慢 29%，说明 FSDP2 在 PPU 上的 all-gather 效率更差。

**评测未跑**（训练未完成）。

## 五、缺什么

| 项 | 原因 |
|---|---|
| reranker 的精度达标判定 | 训练未完成 |
| bge / clip 的吞吐波动区间 | 那两个容器 wandb 当时不可用，日志无每步耗时 |
| 折扣系数 | 需实测芯片 bf16 FLOPS 与显存带宽 |

> reranker 的吞吐波动**已拿到**（wandb `eoxqu2xl`）。

## 六、一句总结

精度三项**几乎与 A100 完全相同**（99.9%-100.1%，是所有平台里最接近的）；
吞吐 BGE 56.8%、CLIP 58.1%，在最严口径下不达标，
但折扣系数只要 ≥1.41 就能全部通过——比昆仑芯的要求（≥1.75）更容易满足。

## 算子级剖析（为吞吐达标做的优化依据）

方法：FlagGems（`github.com/FlagOpen/FlagGems`，1,032 个算子）提供了算子级替换，
所以先要找出**哪些算子真的占时间**。

A100 上用 `torch.profiler` 采三个模型的一次 forward+backward（batch 4），
按 device time 排序；厂商 profiler 不可用（昆仑芯 CUPTI 需要版本相关的环境变量，
试了 `XPU_ENABLE_PROFILER_TRACING` 与 `XPU_CUPTI_ENABLE_DEVICE` 都未生效），
因此**改用逐算子微基准**，三平台跑同一份代码（`/tmp/op_bench.py`）。

### A100 上三模型的算子占比（profiler）

| 模型 | 第一热点 | 占比 | 第二 | 占比 |
|---|---|---|---|---|
| BGE-M3 | bf16 GEMM（各 shape） | **~55%** | elementwise + reduce | ~18% |
| CLIP | bf16 GEMM | ~52% | flash attention | ~12% |
| Reranker-8B | `aten::mm`（LM head 为主） | **36.7%** | bf16 GEMM | ~27% |

**结论：GEMM 是绝对主导（50-64%），其次是 attention 和 elementwise/reduction。**
FlagGems 的融合 kernel 价值在后者——把 `add + rms_norm`、`silu + mul` 这类
多次访存的组合合成一次。

### 三平台逐算子对比（ms/次）

同一份基准代码，A100=1.00 基准。

| 算子 | A100 | 昆仑芯 | 倍数 | PPU | 倍数 |
|---|---|---|---|---|---|
| `mm_lm_head` (4096×1024 @ 1024×151669) | 15.62 | 13.31 | **0.85×** | 13.68 | **0.88×** |
| `mm_ffn` (4096×1024 @ 1024×8192) | 0.352 | 0.636 | 1.8× | 1.054 | 3.0× |
| `mm_hidden` (4096×1024 @ 1024×1024) | 0.092 | 0.104 | 1.1× | 0.638 | **6.9×** |
| `flash_attn_fwd` (B8 S512 GQA) | 1.440 | 2.216 | 1.5× | 3.460 | 2.4× |
| `rms_norm` | 0.024 | 0.141 | **5.9×** | 0.043 | 1.8× |
| `layer_norm` | 0.034 | 0.044 | 1.3× | 0.258 | **7.6×** |
| `silu` | 0.021 | 0.085 | **4.0×** | 0.031 | 1.5× |
| `gelu` | 0.021 | 0.249 | **11.9×** | 0.235 | **11.2×** |
| `add` | 0.017 | 0.033 | 1.9× | 0.025 | 1.5× |
| `mul` | 0.017 | 0.033 | 1.9× | 0.037 | 2.2× |
| `sum_last` | 0.019 | 0.025 | 1.3× | 0.353 | **18.6×** |
| `softmax` | 0.026 | 0.089 | 3.4× | 0.035 | 1.3× |
| `bmm_scores` (B8 128×128) | 0.033 | 0.094 | 2.8× | 0.505 | **15.3×** |
| `cross_entropy` | 0.085 | 0.116 | 1.4× | 0.130 | 1.5× |

### 按优化价值排序（两列分别对应两家）

| 优先级 | 昆仑芯 | PPU |
|---|---|---|
| **P0** | `gelu` **11.9×** | `sum_last` **18.6×** |
| **P0** | `rms_norm` **5.9×** | `bmm_scores` **15.3×** |
| **P1** | `silu` 4.0× | `gelu` **11.2×** |
| **P1** | `softmax` 3.4× | `layer_norm` 7.6× |
| **P2** | `bmm_scores` 2.8× | `mm_hidden` **6.9×** |
| **P2** | `mm_ffn` / `add` / `mul` ~1.8-1.9× | `mm_ffn` 3.0× / `flash_attn` 2.4× |

**PPU 的画像：大 GEMM 强、归约与矩阵乘小算子极弱。**

- ✅ **大 GEMM 比 A100 还快**：`mm_lm_head` 13.68 vs 15.62 ms（**0.88×**）
- ❌ **`sum_last` 慢 18.6×**（0.353 vs 0.019 ms）——最严重
- ❌ **`bmm_scores` 慢 15.3×**（0.505 vs 0.033 ms）——对比学习的核心算子
- ❌ **`gelu` 慢 11.2×`、`layer_norm` 慢 7.6×`、`mm_hidden` 慢 6.9×`

**FlagGems 优化清单（按对训练的预期收益排序）**

| 优先级 | 算子 | 当前倍数 | 理由 |
|---|---|---|---|
| **P0** | `sum` / `mean`（归约） | **18.6×** | 归约路径实现效率极差 |
| **P0** | `bmm` | **15.3×** | CLIP 的 InfoNCE 打分、BGE 的 batch 内负例都靠它 |
| **P1** | `gelu` | **11.2×** | BGE-M3 的 FFN |
| **P1** | `layer_norm` | **7.6×** | CLIP 每层多次 |
| **P2** | `mm_hidden` | 6.9× | 小 GEMM 走了慢路径，大 GEMM 正常 |

> `bmm_scores` 慢 15.3× 与 CLIP 吞吐只有 58.1% 直接相关——CLIP 的对称
> InfoNCE 要对整 batch 做 `text @ image.T`，正是 bmm。

### 这些慢算子在 FlagGems 里有没有？——**有，PPU 也有专属后端**

FlagGems 的 `backends.yaml` 里 **`thead`（平头哥 PPU）也是官方后端**：

| 路径 | 内容 |
|---|---|
| `runtime/backend/_thead/ops/` | **77** 个 PPU 定制算子 |
| `runtime/backend/_thead/fused/` | 2 个 |
| `runtime/backend/_thead/mm_ppu_expand.yaml` | 661 行矩阵乘调优展开 |
| `runtime/backend/_thead/general_ops_thead_expand.yaml` | 通用算子调优参数（block_size / stages / warps 扫描） |
| `runtime/backend/_thead/tune_configs.yaml` | 调优配置 |

逐个核对：

| 算子 | FlagGems 有 | PPU 定制版 | 融合版 |
|---|---|---|---|
| `sum` / 归约 | ✅ `ops/sum.py`（390 行） | — | — |
| `bmm` | ✅ `ops/bmm.py`（223 行） | ✅ `_thead/ops/bmm.py`<br>✅ `baddbmm.py` | — |
| `gelu` | ✅ `ops/gelu.py`（99 行） | — | ✅ `fused/gelu_and_mul.py` |
| `layer_norm` | ✅ `ops/native_layer_norm.py` | — | ✅ `fused/post_layer_norm_residual.py`<br>✅ `fused/skip_layernorm.py` |

**`bmm` 是 PPU 的 P0（慢 15.3×）**，而 `_thead/ops/bmm.py` 是 PPU 专用的实现，
`mm_ppu_expand.yaml` 里还有 661 行的矩阵乘调优展开——**正是针对这类问题做的**。

### 接入的可行性：比昆仑芯好

FlagGems 的 thead 后端声明：

```python
vendor_info = VendorDescriptor(
    vendor_name="thead",
    device_name="cuda",        # PPU 通过 torch.cuda 访问
    dispatch_key=None,         # 用标准 CUDA dispatch key
    device_query_cmd="ppu-smi",
    tle_enabled=True,
)
```

实测这台机器：

| 项 | 状态 |
|---|---|
| `triton` | **3.6.0**（厂商版） |
| `triton.language.extra.cuda` | ✅ **OK** |
| `flag_gems` | ❌ 未安装 |

**不需要额外的 xpu extra**——PPU 走 CUDA 兼容路径，现有的 triton 就带 `extra.cuda`。
所以 PPU 的接入只差 **`pip install flag_gems` + `GEMS_VENDOR=thead`**，
不用等厂商补 Triton。

### 预期收益（按算子占比推算）

CLIP 的 GEMM 占 52%，剩余 48% 里 `bmm` 是核心。若把 `sum`（18.6×）、
`bmm`（15.3×）、`gelu`（11.2×）、`layer_norm`（7.6×）追到 1.5× 以内，
非 GEMM 部分提速约 **2.9×**，整体从 58.1% 到 **~80%**，刚好过门槛。

### 附：fp32 GEMM 与折扣系数

实测（见 `07-a100.md`）表明我们的负载是 **bf16-GEMM**：BGE 的 fp32 GEMM 只占
0.18%（对比学习打分矩阵），CLIP 为 0%，reranker 为 0.001%。

**所以 PPU 的折扣系数应按 bf16 FLOPS + 显存带宽计算**，
不需要把 fp32 FLOPS 纳入——`master_dtype=fp32` 只影响常驻显存与 optimizer，
前向/反向的 GEMM 仍是 bf16。

## 算子级剖析：真实 step 归因

方法同 `08-kunlunxin.md`：在**真实 forward+backward** 上做 profiler，
按语义归类后对比三个平台。逐算子微基准会给出错误排序（见 08 的教训）。

**PPU 的 profiler 有一个坑**：它同时输出 `aten::` 层和底层 kernel 层，
两者相加正好等于报告的总时间——**是同一份工作的重复计数**。
必须只取一层。本文取 `aten::` 层（110.76 ms），底层 kernel 是它的一部分。

### BGE-M3 真实 step（batch 4 × seq 512）

| 类别 | A100 | 昆仑芯 | **PPU** | PPU/A100 |
|---|---|---|---|---|
| **gemm**（`mm`+`addmm`+`bmm`） | 6.73 | 39.12 | **70.19** | **10.43×** |
| other | 2.60 | 27.90 | 16.46 | 6.33× |
| **reduce**（sum/all） | 1.65 | 21.48 | **11.44** | **6.92×** |
| dtype_cast（`_to_copy`） | 0 | 21.28 | 4.54 | — |
| layer_norm | 4.67 | 8.17 | 3.27 | **0.70×（更快）** |
| dropout | 2.42 | 4.20 | 2.85 | 1.18× |
| **gelu** | 2.17 | 38.51 | **2.01** | **0.93×（持平）** |
| flash_attention | 5.01 | 6.30 | — | 被融合进 other |
| transpose | 0 | 8.18 | 0 | — |
| **合计** | **25.25** | **175.14** | **110.76** | **4.39×** |

### PPU 的逐 op 明细（aten 层，前 10）

| op | 调用/步 | ms | 占比 |
|---|---|---|---|
| **`aten::mm`** | 1,444 | **43.73** | **19.3%** |
| **`aten::addmm`** | 725 | **18.65** | **16.8%** |
| `aten::sum` | 725 | 10.21 | 9.2% |
| `aten::add_` | 1,758 | 7.93 | 7.2% |
| `aten::bmm` | 720 | 7.81 | 7.1% |
| `aten::_to_copy` | 1,120 | 4.54 | 4.1% |
| `aten::clone` | 490 | 2.10 | 1.9% |
| `native_layer_norm_backward` | 245 | 2.10 | 1.9% |
| `aten::mul` | 616 | 1.86 | 1.7% |
| `gelu_backward` | 120 | 1.24 | 1.1% |

### 与昆仑芯的关键差异

| | 昆仑芯 | PPU |
|---|---|---|
| **头号问题** | `gelu`（17.77×，+36 ms） | **小 GEMM（`mm`+`addmm` = 62.4 ms，占总 56%）** |
| `gelu` | 17.77× 慢 | **0.93×，持平** |
| `layer_norm` | 1.75× 慢 | **0.70×，更快** |
| 胶水 kernel | 58.31 ms（33%） | 较少（`_to_copy` 4.54 ms） |
| flash attention | 1.26× | 被融合，单独看不出来 |

**两家的短板完全不同**：昆仑芯是激活函数 + 胶水 kernel，PPU 是小 GEMM。

### 优化优先级（PPU）

| 优先级 | 目标 | 每步增量 | FlagGems 能补？ |
|---|---|---|---|
| **P0** | **`mm` + `addmm` + `bmm`**（70.19 ms，占 63.4%） | **+63.5 ms** | ❌ GEMM 不在 FlagGems 范围 |
| P1 | `sum`（10.21 ms） | +9.4 ms | ✅ 有，但昆仑芯实测更慢 |
| P2 | `add` / `_to_copy`（12.47 ms） | +11.0 ms | ✅ 有，实测更慢 |
| — | `gelu` / `layer_norm` | **PPU 上比 A100 快**，不动 | — |

**PPU 的瓶颈是 1,444 次 `mm` 调用 + 725 次 `addmm`**，这些是小矩阵乘
（batch 4 的 encoder），走的是 `gemm_ktype0_aiu1_...` 内核。
这类形状的 GEMM 效率问题要靠厂商的 GEMM 库调优，**不是换 elementwise 算子能解决的**。

> 注：PPU 的折扣系数是 0.711（比昆仑芯的 1.321 低），门槛更松，
> 所以即使慢 4.39×，CLIP 与 reranker 仍然达标、BGE 只差 0.18%
> （见 `10-microbench-discount.md`）。

### 一个查证过的否定项：`mm_ppu_expand.yaml` 帮不上忙

FlagGems 的 `_thead/mm_ppu_expand.yaml` 有 **661 行**矩阵乘调优参数，
按形状分了 18 组（`mm_ppu` / `mm_ppu_small_m` / `mm_ppu_narrow_n` /
`bmm_ppu` / `gemv_ppu` / `mm_ppu_split_k` ...），看起来正对着 PPU 的小 GEMM 问题。

**但它对原生 `aten::mm` 无效。** 这些是 FlagGems **自己 Triton kernel** 的
`BLOCK_M/BLOCK_N/BLOCK_K/num_warps/num_stages` 调参，只在
`with flag_gems.use_gems():` 里替换掉原生实现后才生效。

而实测：

| 事实 | 结果 |
|---|---|
| FlagGems 的 `mm` 在昆仑芯上 | 0.52× （微基准）/ 未测 PPU |
| PPU 真实 step 的 `mm` 调优 | ❌ 走的是 `gemm_ktype0_aiu1_...`，厂商 XBLAS 内核 |

所以 **PPU 的小 GEMM 问题只能靠厂商 XBLAS 解决**，改 FlagGems 的 yaml 没有用。

## 一个必须澄清的落差：吞吐达标 ≠ 每 step 只慢一点

BGE 的吞吐是 **66.6 texts/s，达门槛 99.8%**（门槛 = 118.0 × 0.711 × 0.8 = 67.1），
但 profiler 显示**每 step 的 kernel 时间比 A100 慢 4.39×**。两个数都对，
因为它们量的不是同一件事。

### 分解

BGE 配置 `batch_size 10 × grad_accum 4` = **40 行/step = 200 texts/step**，
共 23,010 step。

| 项 | A100 | PPU | 倍数 |
|---|---|---|---|
| **整 step 墙钟**（训练日志） | **1,696 ms** | **3,004 ms** | **1.77×** |
| 其中 **kernel 时间**（profiler ×10） | 252 ms（**15%**） | 1,108 ms（**37%**） | **4.39×** |
| 其中 **非 kernel 时间** | 1,443 ms（85%） | 1,897 ms（63%） | 1.31× |
| 吞吐 | 117.9 texts/s | 66.6 texts/s | 0.56× |

**`110.76 ms` 是「4 行、20 texts 的一次 forward+backward 的 GPU kernel 时间总和」，
不是 step 时间。** 真实的 PPU step 是 3.0 秒——kernel 只占 37%。

非 kernel 部分是：数据加载与 tokenize/collate、AdamW 更新、H2D 拷贝、
以及每 epoch 的验证与最终测试。

### 为什么 kernel 慢 4.39× 只让吞吐慢 1.77×

因为**两个平台的 step 都被非 kernel 开销主导**（A100 占 85%、PPU 占 63%），
这部分接近共享（A100 1.44 s vs PPU 1.90 s，只差 1.31×），把 kernel 的差距稀释了。

### 这个达标的性质：**是真实的，但很薄**

按 docx 的定义，吞吐 = 训练集总样本数 / 训练总时间，是端到端墙钟——
所以 **99.8% 是合法的达标**，不是错算。

但要清楚它靠什么成立：

| 假设 | PPU 相对 A100 | vs 门槛 56.9% |
|---|---|---|
| 实测（含非 kernel 开销） | 56.6% | ✅ 99.8% |
| 若 kernel 占 100%（纯 kernel 效率） | **22.8%** | ❌ **差 2.5×** |

**PPU 的达标是被"非 kernel 开销占比高"扛过去的**。这意味着：

1. **余量极薄（0.18%）**——任何让 PPU 端到端变慢 1% 的事（比如多一个进程抢卡）
   都会掉出门槛。
2. **如果优化了数据管线**（比如加 worker、上缓存），两个平台的 kernel 占比都会
   上升，PPU 的相对位置会**变差**，因为差距在 kernel 里。
3. **如果增大 batch**（让 kernel 占比上升），同理 PPU 会掉出达标。
4. 所以**不能把"PPU BGE 达标"当成"PPU 的 BGE 算子已经够快"**——
   它在 kernel 层面还差 4.39×，只是被开销掩盖了。

> 反过来看也对：A100 的 85% 非 kernel 说明**这个 workload 的瓶颈其实在数据管线**，
> 不在算力。真正想让三个模型跑得快，数据管线的收益比换算子更大——
> 而且对两个平台同时有效。
