# 20 · 关键慢算子的识别方法与脚本

**给甲方看的两件事**：我们目前用什么方法找慢算子、这个方法有什么局限。
**局限是重点**——它决定了这部分工作应该由运行时层（厂商/编译器团队）接手，
而不是我们自己继续往下挖。

---

## 一、方法总览

三层递进，每层回答一个不同的问题：

| 层 | 问题 | 工具 | 产出 |
|---|---|---|---|
| **1. 阶段拆解** | 慢在计算、还是数据管线？ | `profile_step_segments.py` | 每 step 的 device 计算 / tokenize / dataloader 占比 |
| **2. 算子归因** | 计算里哪个算子占时间？ | `torch.profiler` | 按 kernel 名排序的耗时与调用次数 |
| **3. 差距归因** | 比 A100 多花的时间来自哪里？ | 上面两步的结果相减 | 按"占差距比例"排序的清单 |

**第 3 层是给适配方派活的依据**，不是"慢多少倍"排序——一个慢 5× 但每步只调
1 次的算子，不如一个慢 1.2× 但占 44% 的算子重要。

> ⚠️ **三层方法都在单卡上做。** 多卡并行下 profiler 会失真——2 卡 FSDP2 实测
> 通信占 56.3%，kernel ms 也随卡数变化。详见 §5.5。本文的清单测的是
> "算子自身有多快"，不能用来回答"多卡训练为什么慢"。

---

## 二、第 1 层：阶段拆解

### 2.1 脚本

`examples/retrieval/profile_step_segments.py`

```bash
python examples/retrieval/profile_step_segments.py \
    --conf examples/retrieval/conf/train/bge_m3.yaml \
    --steps 3 --skip 2 --device cuda
```

| 参数 | 说明 |
|---|---|
| `--conf` | 真实配置（不是合成的 batch） |
| `--steps` | 计时步数（默认 3，取中位数）|
| `--skip` | 预热步数（默认 2，不计时）|
| `--prof-steps` | profiler 步数（默认 3，取平均）|
| `--device` | `auto` / `cuda` / `flagos` / `npu` |
| `--out` | 输出 JSON |

### 2.2 三个设计要点（都是踩过坑才有的）

**1. 用真实配置，不用合成输入。**
早期用 batch 4 的合成输入做 profile，得到"flash attention 慢 2.04×，占 BGE 增量 93%"。
真实配置下（batch 10 × accum 4 = 200 texts/step，padded 长度 299）实测
**flash attention 是 0.86×，反而更快**。合成输入的行数和长度分布完全不同，
结论会整个反过来。

**2. 计时和 profile 分两个阶段。**
`torch.profiler` 会让墙钟翻倍。同一步既计时又 profile，测出来的时间没有意义。

**3. tokenize 只在预热步单独测。**
否则同一批数据会被 tokenize 两次，重复计入。

### 2.3 实测结果（BGE-M3）

| 阶段 | A100 | 昆仑芯 | 倍数 |
|---|---|---|---|
| 整 step 墙钟 | 1,605 ms | 3,262 ms | 2.03× |
| **device 计算** | **1,504 ms（93.7%）** | **3,132 ms（96.0%）** | 2.08× |
| tokenize | 107 ms（6.7%） | 105 ms（3.2%） | **0.98×** |
| dataloader `next()` | 0.67 ms（0.04%） | 1.11 ms（0.03%） | — |

**结论：两个平台的 step 都有 94-96% 在 device 计算里。数据管线不是瓶颈**
（dataloader 占 0.04%，tokenize 两边几乎相同）。这一条直接排除了"优化数据加载"
这个方向。

---

## 三、第 2 层：算子归因

### 3.1 脚本

`examples/retrieval/profile_op_micro.py`（逐算子微基准）
+ `torch.profiler`（真实 step 的 kernel 归因，在 `profile_step_segments.py` 的 profile 阶段里）

```bash
python examples/retrieval/profile_op_micro.py --out ops.json --label kunlunxin
```

### 3.2 昆仑芯的两个 profiler 坑

```
XPU_ENABLE_PROFILER_TRACING=1
XPU_CUPTI_ENABLE_DEVICE=0        # 注意是逻辑设备号 0，不是物理卡号
```

另外它的 event key 是**裸 kernel 名**（`xpukernel_xpu3::gelu_bwd_...`），
不是 `aten::` 前缀，过滤条件不能按前缀写。

### 3.3 复现性

两平台各跑两遍，kernel 时间：A100 1,419 / 1,415 ms，昆仑芯 2,923 / 2,923 ms
（**差 <0.3%**）。分段实测（device 计算 1,504 / 3,132 ms）与 profiler 的 kernel 合计
（1,419 / 2,923 ms）相差 6-7%，差异是 profiler 统计不到的同步空隙与 host 侧发射开销。
**两个独立口径互相印证。**

---

## 四、第 3 层：差距归因（最关键）

### 4.1 昆仑芯 BGE-M3 的归因

**基线**：整步 kernel 2,923 ms（A100 1,419 ms，2.06×），差距 **+1,503 ms**。

| # | 算子/类别 | 每步耗时 | 慢多少 | **占差距** | 具体 kernel |
|---|---|---|---|---|---|
| **1** | **GELU** | **1,028.9 ms**（35.2%）| **25.4×** | **65.8%** | `gelu_bwd_highprecision_nv` 480 次 654.2 ms<br>`gelu_highprecision_nv` 480 次 374.7 ms |
| **2** | **cast/transpose 胶水** | **309.0 ms**（10.6%）| A100 无此开销 | **20.6%** | `cast_te` 4,800 次<br>`cluster_cast_kl3` 12,399 次<br>`transpose_021_sdnn` 1,936 次 |
| **3** | GEMM | 819.8 ms（28.1%）| 1.2× | 10.2% | `fc_bf16_rn_*` |
| — | attention | 277.4 ms | **0.86×（更快）** | — | `flash_attn_bwd_xpu3` |
| — | layer_norm | 61.2 ms | **0.80×（更快）** | — | — |

**结论：差距的 66% 来自 GELU 一项，21% 来自 dtype 转换/转置胶水。
GEMM 只占 10%，attention 和 layer_norm 在昆仑芯上反而更快。**

### 4.2 GELU 为什么慢 25 倍

**机制**：`aten::gelu` 在 autocast 的 fp32 白名单里，所以**即使模型是 bf16，
GELU 也走厂商的 `_highprecision` 路径**（fp32 计算）。

| 事实 | 数值 |
|---|---|
| 占总 kernel 时间 | 35.2% |
| 比 A100 慢 | 25.4×（A100 的 `aten::gelu`+`gelu_backward` 只有 40.5 ms）|
| 成本随行数 | **线性 0.64 µs/行**（A100 在 128-1024 行是恒定 0.12 ms，启动开销主导）|

### 4.3 PPU 的归因（短板完全不同）

**基线**：整步 kernel 2,531 ms（A100 的 1.78×），差距 **+1,111 ms**。

| # | 类别 | 每步 | 慢多少 | 占差距 |
|---|---|---|---|---|
| **1** | **GEMM** | 1,158.6 ms | 1.6-1.8× | **44.3%** |
| **2** | attention（无融合内核）| 553.5 ms | 1.7× | **20.4%** |
| 3 | copy/cast | 248.8 ms | 2.9× | 14.7% |
| — | **GELU** | 36.8 ms | **0.9×（更快）** | — |
| — | layer_norm | 45.2 ms | **0.6×（更快）** | — |

**同一个算子在不同芯片上结论相反**：昆仑芯的 GELU 是灾难（25×），PPU 的 GELU
比 A100 还快（0.9×）；而 PPU 的 GEMM 是主要瓶颈（44%），昆仑芯的 GEMM 只占 10%。

> **这说明慢算子清单必须按芯片分别出，不能共用一张表。**

---

## 五、⚠️ 方法的局限（这部分请甲方重点看）

**我们的方法比较 naive，已经挖到它能挖的极限。** 具体四条：

### 5.1 只能"发现"，不能"定位根因"

profiler 告诉我们 GELU 花了 1,028.9 ms、比 A100 慢 25.4×，
但**为什么慢**——是 fp32 转换？是向量化没做？是访存模式？——我们看不到。

这些答案在 kernel 实现里，属于**运行时层的工作**。我们能做的只是把
"哪个 kernel、调用多少次、花了多久、占总差距多少"测准，然后交出去。

### 5.2 需要为每个芯片单独适配 profiler

| 平台 | 特殊要求 |
|---|---|
| 昆仑芯 | `XPU_ENABLE_PROFILER_TRACING=1` + `XPU_CUPTI_ENABLE_DEVICE=0`；event key 是裸 kernel 名 |
| PPU | 未单独验证（profiler 可用性未测）|
| 昇腾 | 未验证 |

**每接一款新芯片，profiler 本身要先调通**，这是一次性成本但不小。

### 5.3 归因依赖"两边 kernel 名能对上"

按语义归类（把 `gelu_bwd_highprecision_nv` 归到 GELU）需要人工建映射表。
如果厂商改名或融合了两个算子，归类就会错。

**典型例子**：昆仑芯的 `cast_te` / `cluster_cast_kl3` 在 A100 上根本不存在
（被融合掉了），这类"只在一边存在的算子"只能单独列一类，无法按倍数对比。

### 5.4 单步波动大，必须多步取平均

同一配置相邻两步的耗时能差 25%（因为每步的 padded 长度不同，取决于该 batch
的最长样本）。我们用"计时 3 步取中位数 + profile 3-4 步取平均"，
但**这仍是采样，不是精确分解**。

### 5.5 ⚠️ 多卡并行下 profiler 会失真（2026-11-09 实测）

**这一条是最重要的局限，之前的版本漏了。**

我们的 profiler 脚本是**单进程**的（`init_distributed` 只用来拿设备，
从没在多卡上跑过）。为回答"多卡下会怎样"，做了对照实验：同一份 BGE-M3，
单卡 vs 2 卡 FSDP2，各 profile 一步。

| | 单卡 | **2 卡 FSDP2（rank0）** |
|---|---|---|
| 总 device 时间 | 290.6 ms | **664.9 ms（2.3×）** |
| kernel 数 | 104 | 114 |
| **通信占比** | 0% | **56.3%** |
| 第一大项 | `fmha_cutlassB` 33.5 ms × 120 | **`ncclDevKernel_AllGather_RING` 308.6 ms × 246** |
| 第二大项 | `ampere_bf16_s16816gemm` 29.9 ms × 360 | `ncclDevKernel_ReduceScatter` 65.6 ms × 25 |

**四个具体问题**：

**(1) 通信和计算混在一张表里。**
`AllGather_RING` 308.6 ms 比任何真实算子都大。如果按"总时间排序"派活，
第一个优化目标会变成"优化 NCCL"——完全错误的方向。

**(2) kernel 耗时会随卡数变化。**
`fmha_cutlassB` 单卡 33.5 ms → 2 卡 18.2 ms（**减半**，每卡只算一半数据）。
所以 **kernel ms 不能跨并行度比较**，只能同配置对比。

**(3) 集合通信的耗时含"等别人"。**
Ring all-gather 里快的 rank 等慢的，profiler 报的是**墙钟等待**，
不是通信带宽。要判断通信是否真慢，得看它占整步的比例 + 卡间是否同步。

**(4) PP 下更严重：每个 stage 只持一部分层。**
Megatron PP=4 时 rank0 只有 9 层，profiler 看到的是 **9 层的算子 +
大量 p2p send/recv 等待**，而真正的瓶颈可能在另一个 stage。

**所以多卡场景下的正确做法**：

| 要回答的问题 | 该怎么做 |
|---|---|
| 这个 kernel 本身多快？ | **单卡跑**（我们的清单就是这么来的）|
| 并行引入多少通信开销？ | profile 后**先把集合通信单列一类**，再看剩余 |
| 整体瓶颈在计算还是通信？ | 用阶段拆解（§二）比 profiler 更可靠——它直接测墙钟 |
| PP 下哪个 stage 慢？ | **每个 rank 各 profile 一次**并对比，不能只看 rank0 |

> **我们目前的慢算子清单（[13](13-bottleneck-ops.md)）全部来自单卡 profile**，
> 这个口径是对的——它测的是"算子自身有多快"。但**它不能回答"多卡训练为什么慢"**，
> 那个问题需要另一套方法（通信 vs 计算的时间分解），我们没做。

### 5.6 结论：建议由运行时层接手

| 我们已完成 | 建议移交 |
|---|---|
| 定位到具体 kernel（名字、调用次数、耗时）| 分析 kernel 内部为什么慢 |
| 算出每个算子占差距的比例 | 决定优化哪个（成本/收益）|
| 排除非瓶颈（GEMM/attention/layer_norm 在这些场景不是问题）| 实现融合、向量化、精度路径修正 |
| 给出 FlagGems 的实测结论（**两平台都更慢，不是解法**）| 提供原生 bf16 kernel |

**我们要的不是"帮我们优化代码"，是"这个 kernel 在你们的运行时里为什么是这个性能"。**

---

## 六、已产出的清单（可直接派活）

| 文档 | 内容 |
|---|---|
| [13-bottleneck-ops](../analysis/13-bottleneck-ops.md) | **按"占差距比例"排序的完整清单**，含 kernel 名、每步调用次数、耗时 |
| [11-vendor-questions](../analysis/11-vendor-questions.md) | 给厂商的问题清单：量化后的瓶颈 + 复现脚本 + 已排除方向 |
| [12-flaggems-integration](../training/12-flaggems-integration.md) | FlagGems 实测结论：**两平台都不快，不是解法** |

### 6.1 昆仑芯：前 3 项占差距 96.5%

| # | 项 | 占差距 | 建议给谁 |
|---|---|---|---|
| 1 | GELU（bf16 原生、向量化）| 65.8% | 运行时层 |
| 2 | cast/transpose 胶水（kernel 融合）| 20.6% | 运行时层 |
| 3 | GEMM（1.2×，收益有限）| 10.2% | 优先级低 |

**要适配方做什么**：提供 bf16 原生、向量化的 `gelu` / `gelu_backward`，
**不要走 fp32 高精度路径**。目标是把 1,028.9 ms 降到 150 ms 以内（约 7×），
BGE 吞吐能从 54.0 提到约 62 样本/s。

### 6.2 PPU：前 2 项占差距 64.7%

| # | 项 | 占差距 | 建议给谁 |
|---|---|---|---|
| 1 | GEMM（中小 M 形状效率）| 44.3% | 运行时层 |
| 2 | attention 无融合内核（SDPA 落到 math 后端）| 20.4% | 运行时层 |

**关键度量**：PPU 的 `mm` 调用次数**与 A100 完全相同**（5,776 次），
但 2990×1024×1024 只有 **44 TFLOP/s**、2990×1024×4096 有 102 TFLOP/s。
**是"同等调用次数下每次更慢"，不是"小 GEMM 调用过多"。**

---

## 七、复现

```bash
# 阶段拆解
python examples/retrieval/profile_step_segments.py \
    --conf examples/retrieval/conf/train/bge_m3.yaml \
    --steps 3 --skip 2 --prof-steps 4 --device cuda \
    --out segments_a100.json

# 逐算子微基准
python examples/retrieval/profile_op_micro.py --out ops_a100.json --label a100
```

**同一份 conf、同一步数、同一统计方法**跑两边，结果才可比。
