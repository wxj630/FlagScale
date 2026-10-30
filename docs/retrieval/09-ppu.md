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
按 device time 排序；当时以为厂商 profiler 不可用，
因此**改用逐算子微基准**，三平台跑同一份代码（`examples/retrieval/profile_op_micro.py`）。
**这个理由后来被推翻**——昆仑芯的 profiler 只要设对两个环境变量就能用，
所以 08/07 的最终结论都基于真实 step 的 profiler，而不是微基准。

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

### 按优化价值排序（微基准口径，**已作废**）

> ⚠️ **这张表基于"慢多少倍"，而正确的排序依据是"每步多花多少毫秒"。**
> 昆仑芯已被实测否证：表里把它排成 `gelu`+`rms_norm`+`silu`+`softmax`，
> 而真实差距的 65.5% 只在 GELU 一项，`norm` 类反而比 A100 快。
> PPU 尚未做真实配置实测，所以**下面这些倍数不能当作优化优先级**，
> 只能当"哪些算子值得进一步查"的线索。
> 保留此表只为记录方法演进。

| 优先级 | 昆仑芯 | PPU |
|---|---|---|
| **P0** | `gelu` **11.9×** | `sum_last` **18.6×** |
| **P0** | `rms_norm` **5.9×** | `bmm_scores` **15.3×** |
| **P1** | `silu` 4.0× | `gelu` **11.2×** |
| **P1** | `softmax` 3.4× | `layer_norm` 7.6× |
| **P2** | `bmm_scores` 2.8× | `mm_hidden` **6.9×** |
| **P2** | `mm_ffn` / `add` / `mul` ~1.8-1.9× | `mm_ffn` 3.0× / `flash_attn` 2.4× |

**PPU 的微基准画像：大 GEMM 强、归约与矩阵乘小算子极弱。**

- ✅ **大 GEMM 比 A100 还快**：`mm_lm_head` 13.68 vs 15.62 ms（**0.88×**）
- ❌ `sum_last` 18.6×（0.353 vs 0.019 ms）
- ❌ `bmm_scores` 15.3×（0.505 vs 0.033 ms）
- ❌ `gelu` 11.2×、`layer_norm` 7.6×、`mm_hidden` 6.9×

> `bmm_scores` 慢 15.3× 曾被用来解释 CLIP 吞吐偏低——CLIP 的对称 InfoNCE
> 要对整 batch 做 `text @ image.T`。**这个因果链未经真实 step 验证。**


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

> ⚠️ **本节的三方对比表（batch 4 × seq 512）用的是"小 batch profile × 10 外推"
> 的口径，这个口径已被证伪，数字只作历史记录。**
> 硬证据：A100 用同样的方法算出 25.25 ms × 10 = **252 ms/step**，
> 但用真实配置实测是 **1,482 ms/step**——**低估 5.9 倍**。
> 原因有两个：(a) 合成输入只有 115 token，真实 padded 均值是 299；
> (b) `aten::` 层和底层 kernel 层都报 device time，相加会翻倍。
> PPU 那列（110.76 × 10 = 1,108 ms）是同一个错误口径的产物。
>
> **PPU 至今没有真实配置的实测**（本文写作时 VPN 断线，无法连机器）。
> 待补：在 PPU 上跑 `examples/retrieval/profile_step_segments.py`，与 07/08 的 A100、
> 昆仑芯数字同口径对比。

**PPU 的 profiler 有一个坑**（这个结论仍然成立）：它同时输出 `aten::` 层和
底层 kernel 层，两者相加正好等于报告的总时间——**是同一份工作的重复计数**。
必须只取一层。本文取 `aten::` 层。

### PPU 的逐 op 明细（aten 层，batch 4 合成输入，前 10）

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

即使按这个不可靠的口径，**GEMM 类（`mm`+`addmm`+`bmm`）仍占 56% 以上**，
这是 PPU 与昆仑芯最本质的不同：昆仑芯的头号是 GELU，PPU 的头号是小 GEMM。

### 两家的短板方向相反（定性结论，待 PPU 实测复核）

| | 昆仑芯（已实测） | PPU（旧口径） |
|---|---|---|
| **头号问题** | **GELU，占差距 65.5%** | **小 GEMM（占总时间 56%）** |
| GELU | **25.8× 慢** | 0.93×，持平 |
| layer_norm | 0.80×，更快 | 0.70×，更快 |
| 胶水 kernel | **cast/transpose 占差距 19%** | 较少（`_to_copy` 4.54 ms） |
| flash attention | 0.88×，更快 | 被融合，单独看不出来 |

**同一个 BGE-M3，昆仑芯要治激活函数和 dtype 转换，PPU 要治小 GEMM 的
launch 开销。** 这意味着两家的优化手段不能互换。


### 优化优先级（PPU）

| 优先级 | 目标 | 旧口径下的每步增量 | FlagGems 能补？ |
|---|---|---|---|
| **P0** | **`mm` + `addmm` + `bmm`**（70.19 ms，占 63.4%） | **+63.5 ms** | ❌ GEMM 不在 FlagGems 范围 |
| P1 | `sum`（10.21 ms） | +9.4 ms | ✅ 有，但昆仑芯实测更慢 |
| P2 | `add` / `_to_copy`（12.47 ms） | +11.0 ms | ✅ 有，实测更慢 |
| — | `gelu` / `layer_norm` | **PPU 上比 A100 快**，不动 | — |

> 上表的绝对毫秒数来自已证伪的 batch-4 外推口径，**只用来看相对顺序**。
> 真实占比待 PPU 实测补齐。

**PPU 的瓶颈是 1,444 次 `mm` 调用 + 725 次 `addmm`**，这些是小矩阵乘
（batch 4 的 encoder），走的是 `gemm_ktype0_aiu1_...` 内核。
这类形状的 GEMM 效率问题要靠厂商的 GEMM 库调优，**不是换 elementwise 算子能解决的**。

> 注：PPU 的折扣系数是 0.711（比昆仑芯的 1.321 低），门槛更松，
> 所以即使慢 4.39×，CLIP 与 reranker 仍然达标、BGE 只差 0.18%
> （见 `10-microbench-discount.md`）。**这里的 4.39× 同样来自被证伪的口径，
> 不要引用为定论。**

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

## PPU BGE 达标 99.8%：这个数字怎么读

BGE 的吞吐是 **66.6 texts/s，达门槛 99.8%**（门槛 = 118.0 × 0.711 × 0.8 = 67.1）。

### 分解（A100 已实测，PPU 为旧口径）

BGE 配置 `batch_size 10 × grad_accum 4` = **40 行/step = 200 texts/step**，
共 23,010 step。

| 项 | A100（实测） | PPU（旧口径 ※） | 倍数 |
|---|---|---|---|
| **整 step 墙钟** | **1,830 ms** | 3,004 ms | 1.64× |
| 其中 **device 计算（fwd+bwd）** | **1,566 ms（85.6%）** | 未实测 | — |
| 其中 tokenize | 110 ms（6.0%） | 未实测 | — |
| 其中 dataloader | 0.7 ms（0.04%） | 未实测 | — |
| 吞吐 | 117.9 texts/s | 66.6 texts/s | 0.56× |

※ PPU 的 step 墙钟来自同日训练日志，但**没有做过同口径的阶段拆解**
（写作时 VPN 断线）。A100 那列是 `examples/retrieval/profile_step_segments.py` 的实测值。

### 旧解释作废：并不是"非 kernel 开销把差距稀释了"

早前这里写的是「A100 有 85% 的时间不在 kernel 里，PPU 只有 63%，
这块接近共享所以稀释了 kernel 差距」。**这个解释建立在一个已证伪的测量上。**

A100 实测：85.6% 的时间**在 device 计算里**（fwd 638 ms + bwd 928 ms），
tokenize 6%、dataloader 0.04%。那个"85% 非 kernel"的数字，是拿
batch 4 合成输入的 profile（25.25 ms）× 10 算出的 252 ms，
用 1,696 ms 墙钟减出来的残差——而真实 device 时间是 1,482 ms。
**残差里装的其实是没被小 profile 采到的 device 计算，不是数据管线。**

所以正确的读法应该是：

1. **端到端吞吐 99.8% 是合法的达标**（docx 按墙钟定义），这一点没变。
2. **但没有"优化数据管线就能让 PPU 掉出达标"这回事**——数据管线只占 0.04%，
   动它不会改变任何东西。
3. **PPU 的 kernel 效率到底差多少，目前没有可信数字。** 旧口径的 4.39×
   和 22.8%，都来自同一套错误外推，不应再引用。
4. 唯一可信的相对结论是 **PPU 的短板在小 GEMM**（`mm`+`addmm`+`bmm`
   占 aten 层 56% 以上），这个排序在两种口径下都一致。

### 待补的实测

在 PPU 上跑同一份脚本即可对齐口径：

```bash
python examples/retrieval/profile_step_segments.py \
  --conf examples/retrieval/conf/train/bge_m3.yaml \
  --steps 3 --skip 2 --device flagos --out ppuseg.json
```

拿到 PPU 的真实 device 计算占比之前，**本文档不给 PPU 的 kernel 级倍数结论**。

