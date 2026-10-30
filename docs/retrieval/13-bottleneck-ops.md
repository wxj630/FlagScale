# 13 · 拖累算子清单（交给算子适配方）

**这份清单是给"要去做算子适配的人"看的。请按它派活，不要按微基准的倍数派。**

## 零、先回答一个前置问题：FlagGems 能解决吗？

**不能。两个平台都实测过了。**

| 平台 | flag_gems 版本 | 逐算子实测 | 真实训练 |
|---|---|---|---|
| 昆仑芯 P800 | 5.4.0（vendor `kunlunxin`） | **10 个里 9 个更慢**（`sum` 0.03×、`layer_norm` 0.04×、GELU 0.65×） | ❌ 跑不完（3 个 bug） |
| PPU ZW810E | 5.4.0（vendor `thead`） | **13 个里 12 个更慢，GEMM 持平**（0.99×） | ⚠️ 未测 |

细节见 [12-flaggems-integration.md](12-flaggems-integration.md)。
**所以下面这些算子必须由厂商或适配方在原生 kernel 层面解决，换 FlagGems 无效。**

## 一、口径（为什么这份清单可信）

清单**不是**从微基准来的，而是从**真实配置的整步 profiler 归因**来的：

- 真实 conf：BGE-M3，`batch_size 10 × grad_accum 4` = **200 texts/step**
- 实测 padded 长度均值 **299**（不是配置上限 512）
- 计时不开 profiler（开了墙钟翻倍），profile 连跑 4 步取平均
- 两平台各跑两遍，kernel 时间复现误差 **<0.3%**
- 脚本：`examples/retrieval/profile_step_segments.py`

**为什么不用微基准排优先级**：同一算子在不同设备状态下能差 2.15 倍
（PPU 的 `mm_hidden` 就被误测成 1.49× 加速），小 batch 外推更是会把
占比算反（详见 [08](08-kunlunxin.md) 的教训）。**只有整步实测能定优先级。**

排序依据 = **该项占"与 A100 差距"的比例**，不是"慢多少倍"。
一个慢 5× 但每步只调 1 次的算子，不如一个慢 1.2× 但占 44% 的算子重要。

---

## 二、昆仑芯 P800：前 3 项占差距的 96.5%

**基线**：整步 kernel 2,923 ms（A100 1,419 ms，**2.06×**），差距 **+1,503 ms**。

| # | 算子 / 类别 | 每步耗时 | 慢多少 | 占差距 | 具体 kernel / 说明 |
|---|---|---|---|---|---|
| **1** | **GELU** | **1,028.9 ms**（35.2%） | **25.4×** | **65.8%**<br>+988.5 ms | `gelu_bwd_highprecision_nv` 480 次/步 **654.2 ms**<br>`gelu_highprecision_nv` 480 次/步 **374.7 ms** |
| **2** | **dtype cast / transpose** | **309.0 ms**（10.6%） | A100 无此开销 | **20.6%**<br>+309.0 ms | `cast_te` 4,800 次/步<br>`cluster_cast_kl3` 12,399 次/步<br>`transpose_021_sdnn` 1,936 次/步 |
| **3** | **GEMM** | 819.8 ms（28.1%） | 1.2× | 10.2%<br>+153.2 ms | `fc_bf16_rn_*` 系列（多种 algo） |
| 4 | reduce | 91.5 ms | 1.8× | 2.7% | `reduce_tn`、`findmax1d_cluster` |
| 5 | dropout | 57.4 ms | 2.7× | 2.4% | `dropout_sdnn`、`dropout_rand_bwd_v2` |
| 6 | elementwise | 152.1 ms | 1.3× | 2.4% | `tensor_calc_basic`、`tensor_calc_k_nk` |
| — | attention | 277.4 ms | **0.86×（更快）** | — | `flash_attn_bwd_xpu3` / `flash_attention_fwd_2_sdnn` |
| — | layer_norm | 61.2 ms | **0.80×（更快）** | — | `layernorm_bwd_*` |
| — | embedding | 21.1 ms | 2.9× | 0.9% | — |

### 第 1 项：GELU 的细节（这是唯一的大头）

| 事实 | 数值 |
|---|---|
| 占总 kernel 时间 | **35.2%** |
| 比 A100 慢 | **25.4×**（A100 的 `aten::gelu`+`gelu_backward` 合计 40.5 ms） |
| 成本随行数变化 | **线性 0.64 µs/行**（A100 在 128-1024 行是恒定 0.12 ms，属启动开销主导） |
| 现状 | 走了 `_highprecision` 路径（`aten::gelu` 在 autocast 的 fp32 白名单里，bf16 模型也算 fp32） |
| 已知可选项 | `approximate="tanh"` 在昆仑芯快 **1.65×**（1.951→1.162 ms @2990 行），A100 无差别 |
| FlagGems | ❌ 更慢 1.54×（0.52→0.80 ms），且反向触发 `error 719` |

**要适配方做什么**：提供 bf16 原生、向量化的 `gelu` 与 `gelu_backward`，
不要走 fp32 高精度路径。目标是把 1,028.9 ms 降到 150 ms 以内（约 7×），
这样昆仑芯 BGE 吞吐从 54.0 → 约 62 samples/s。

### 第 2 项：cast / transpose 胶水

A100 上这些是被融合掉或成本可忽略的，昆仑芯上是独立的 kernel，共 **19,135 次/步**
（4,800 + 12,399 + 1,936）。
**要适配方做什么**：kernel 融合（把 layernorm 前后的 dtype 转换、
attention 的 transpose 融进主 kernel），或减少不必要的布局转换。

### 第 3 项：GEMM 不是主要问题

只慢 1.2×，占差距 10.2%。**不要在昆仑芯的 GEMM 上花力气**——
微基准里它甚至比 A100 快（bf16 GEMM 实测是 A100 的 127%）。

---

## 三、PPU ZW810E：前 2 项占差距的 64.7%

**基线**：整步 kernel 2,531 ms（A100 1,419 ms，**1.78×**），差距 **+1,111 ms**。

| # | 算子 / 类别 | 每步耗时 | 慢多少 | 占差距 | 具体 op / 说明 |
|---|---|---|---|---|---|
| **1** | **GEMM**（`mm`+`addmm`） | **1,158.6 ms**（45.8%） | **1.6-1.8×** | **44.3%**<br>+491.9 ms | `aten::mm` 5,776 次 **813.1 ms**（1.8×）<br>`aten::addmm` 2,900 次 **345.5 ms**（1.6×） |
| **2** | **attention**（无融合内核） | **553.5 ms**（21.9%） | 1.7× | **20.4%**<br>+226.8 ms | `aten::bmm` 2,880 次 **347.6 ms**（A100 无此调用）<br>`_softmax`+`_softmax_backward_data` **121.3 ms**<br>`where` 53.5 / `isneginf` 31.1 |
| **3** | copy/cast | 248.8 ms（9.8%） | 2.9× | 14.7% | `_to_copy` **13,766 次/步** 166.5 ms<br>`contiguous` 2,468 次 47.7 ms<br>`clone` 1,980 次 34.2 ms |
| **4** | elementwise | 254.5 ms（10.1%） | 2.3× | 12.9% | `add_` 9,358 次 133.4 ms<br>`mul` 2,465 次 87.4 ms<br>`where` 500 次 53.5 ms |
| **5** | dropout | 123.3 ms（4.9%） | **5.9×** | 9.2% | `native_dropout` 1,460 次 62.9 ms（4.6×）<br>`native_dropout_backward` 60.4 ms（8.2×） |
| — | GELU | 36.8 ms | **0.9×（更快）** | — | 不用动 |
| — | layer_norm | 45.2 ms | **0.6×（更快）** | — | 不用动 |
| — | reduce | 54.5 ms | 1.1× | 0.3% | 不是问题 |

### 第 1 项：GEMM 的细节（关键：不是"调用太多"）

| 事实 | 数值 |
|---|---|
| 占差距 | **44.3%**（+491.9 ms） |
| 慢多少 | `mm` 1.8×、`addmm` 1.6× |
| **调用次数** | **与 A100 完全相同**：`mm` 5,776 次、`addmm` 2,900 次 |
| 走的内核 | `gemm_ktype0_aiu1_mtype1_dtypeBF16xBF16xFP32xBF16xBF16_tile*`（厂商 XBLAS） |
| 形状效率 | 2990×1024×1024 只有 **44 TFLOP/s**；2990×1024×4096 有 102 TFLOP/s |

**所以这是"同等调用次数下每次更慢"，不是"小 GEMM 调用过多"。**
早前"PPU 小 GEMM 调用过多"的说法是错的，已修正。

**要适配方做什么**：提升中小 M（1280-5980）形状下 bf16 GEMM 的效率。
当前的 44-102 TFLOP/s 相比大 GEMM 明显偏低。

### 第 2 项：attention 没有融合内核

A100 上是 `_efficient_attention_forward/backward`（一对融合 kernel），
**PPU 上完全没有这个算子**，SDPA 落到 math 后端，展开成
`bmm`+`softmax`+`isneginf`+`where`。

**但有一个必须说清的限制**：单独测前向时，math 路径**并不比 SDPA 慢**
（B=10,H=16,S=299,D=64：sdpa 0.24 ms，math 0.21 ms，同一量级）。
成本集中在：
- **反向 softmax**：`_softmax_backward_data` 67.9 ms/步
- **掩码准备**：`isneginf` 31.1 + `where` 53.5 ms/步

**要适配方做什么**：提供 fused SDPA（含 dropout 支持，见下），
或至少把 mask 处理融进 attention。

---

## 四、给适配方的三条注意事项

1. **两家的名单几乎不重叠，不要合并派活。**
   昆仑芯要治 **GELU + dtype 转换**；PPU 要治 **GEMM + fused attention**。
   在昆仑芯上优化 GEMM、或在 PPU 上优化 GELU，都是浪费。

2. **不要按"慢多少倍"排序。**
   昆仑芯的 `transpose_021_sdnn` 单看倍数是"无穷倍"（A100 上不存在），
   但它只占 2.3%；而 GEMM 只慢 1.2× 却占 10.2%。
   **按占差距的比例排序，本清单已经排好了。**

3. **PPU 的 dropout 慢 4.6-8.2× 值得单独提**：
   占差距 9.2%，且是标准算子，适配成本低、收益明确。

---

## 五、复现

```bash
# 真实 step 归因（两平台同一脚本）
python examples/retrieval/profile_step_segments.py \
    --conf examples/retrieval/conf/train/bge_m3.yaml \
    --steps 6 --skip 4 --prof-steps 4 --device cuda --out seg.json

# Kubernetes：昆仑芯需要
export XPU_ENABLE_PROFILER_TRACING=1 XPU_CUPTI_ENABLE_DEVICE=0

# PPU 用 --device flagos；PPU 的 profiler 会同时输出 aten 层与 kernel 层，
# 相加等于总时间（重复计数），只取一层。
```

每个数字的完整明细（含全部 kernel 名与调用次数）见
[08-kunlunxin.md](08-kunlunxin.md) 与 [09-ppu.md](09-ppu.md) 的
「算子级剖析」章节。
