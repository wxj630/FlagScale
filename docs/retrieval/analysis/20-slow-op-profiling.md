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

> ⚠️ **三层方法都在单卡上做**，测的是"算子自身有多快"——这是可跨芯片比较的
> 唯一口径。多卡（FSDP2 / Megatron PP）下 profile 结果会多出一类集合通信条目，
> 需要单列。**这块建议交给运行时层**，见 §5.5。

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

### 5.2 四个平台的 profiler 可用性（实测状态不同）

| 平台 | 整步 profiler | 微基准 | 特殊要求 |
|---|---|---|---|
| **A100** | ✅ 有 | ✅ 有 | 无，标准 `torch.profiler` |
| **昆仑芯 P800** | ✅ 有 | ✅ 有 | `XPU_ENABLE_PROFILER_TRACING=1` + `XPU_CUPTI_ENABLE_DEVICE=0`（逻辑设备号 0，不是物理卡号）；event key 是**裸 kernel 名**（`xpukernel_xpu3::gelu_bwd_...`），不是 `aten::` 前缀 |
| **PPU ZW810E** | ✅ 有 | ✅ 有 | `profile_step_segments.py` 直接可用。**坑：同时输出 `aten::` 层和底层 kernel 层，两者相加正好等于总时间——是同一份工作重复计数，必须只取一层**（我们取 `aten::` 层）|
| **昇腾 910C** | ❌ **只有微基准** | ✅ 有 | `profile_op_micro_npu.py` 产出的 `op-micro/ascend_910c.json` |

**三个平台有整步 profiler，一个没有**：

- A100 / 昆仑芯 / PPU 的慢算子结论都来自**真实 step 的 profiler**（权重可信）
- **昇腾只有微基准**，`16-ascend.md` 自己写明了这一点：

  > 这是**微基准**，不是整步 profiler；它用来找候选慢算子，不能直接按倍数排优先级。
  > 真实 step 的权重仍需用 `profile_step_segments.py` 在 NPU 上补测。

  所以「RMSNorm 6.5×」这类数字**是候选信号，不是最终权重**——
  要变成"占差距的百分比"还需要 NPU 上的整步测量。

> **每接一款新芯片，profiler 要先调通**，这是真实存在的一次性成本：
> 昆仑芯要两个环境变量且键名不是 `aten::`；昇腾的 NPU profiler 还没接
> （目前只有微基准）。

### 5.3 归因依赖"两边 kernel 名能对上"

按语义归类（把 `gelu_bwd_highprecision_nv` 归到 GELU）需要人工建映射表。
如果厂商改名或融合了两个算子，归类就会错。

**典型例子**：昆仑芯的 `cast_te` / `cluster_cast_kl3` 在 A100 上根本不存在
（被融合掉了），这类"只在一边存在的算子"只能单独列一类，无法按倍数对比。

### 5.4 单步波动大，必须多步取平均

同一配置相邻两步的耗时能差 25%（因为每步的 padded 长度不同，取决于该 batch
的最长样本）。我们用"计时 3 步取中位数 + profile 3-4 步取平均"，
但**这仍是采样，不是精确分解**。

### 5.5 ⚠️ 多卡并行下，profile 结果多一块"通信开销"（建议移交运行时层）

**这是方法的一条边界，不是缺陷——我们只需要指出它，实现应该在运行时层做。**

我们的 profiler 脚本是**单进程单卡**的，所有慢算子清单（GELU 65.8%、
PPU GEMM 44.3%）都是这个口径。**单卡口径测的是"算子自身有多快"，这是对的**，
也是可跨芯片比较的唯一口径。

但一旦上多卡（FSDP2 / Megatron PP），**profile 结果里会多出一类条目：集合通信**。

实测（同一份 BGE-M3，2 卡 FSDP2，只 profile 一步）：

| | 单卡 | 2 卡 FSDP2（rank0）|
|---|---|---|
| 总 device 时间 | 290.6 ms | 664.9 ms |
| **通信占比** | 0% | **56.3%** |
| 第一大项 | `fmha_cutlassB` 33.5 ms | **`ncclDevKernel_AllGather_RING` 308.6 ms** |
| 第二大项 | `ampere_bf16_s16816gemm` 29.9 ms | `ncclDevKernel_ReduceScatter` 65.6 ms |

**这个对比不能当成"多卡让我们变慢了"**——别人也多卡，也付通信代价。
它只说明一件事：**多卡 profile 的表里混了两类东西，必须分开**。

#### 需要运行时层做什么

| 需求 | 说明 |
|---|---|
| **通信与计算分列** | 输出里把集合通信（all-gather / reduce-scatter / p2p）单列一类，不要和算子混排 |
| **按 rank 归集** | PP 下每个 stage 只持一部分层，只看 rank0 会漏掉真正的瓶颈 |
| **区分"等待"与"带宽"** | Ring all-gather 测到的是墙钟停滞（快的 rank 等慢的），不是通信带宽。这两个要分开报 |
| **同并行度对比** | kernel ms 随卡数变化（`fmha_cutlassB` 单卡 33.5 → 2 卡 18.2 ms，因为每卡只算一半），所以**跨并行度不可比，只能同配置对比** |

#### 我们的位置

**我们不做这件事**，原因：

1. 这是 **profiler 工具本身的能力**，不是应用层代码能可靠实现的
2. 我们已经能给出"算子自身多快"（单卡口径），这是派活给适配方的依据
3. 多卡通信效率属于**运行时/通信库层**（NCCL 替代品、集合通信实现），
   应用层调不出来

**如果甲方需要"多卡训练为什么慢"的归因**，那是一个独立课题，
需要通信与计算的时间分解（比如每个集合通信前后插事件、按 step 汇总），
建议一并交给运行时层，用他们自己的 profiler 做——他们能拿到 kernel 内部
和通信库内部的信息，我们拿不到。

### 5.6 结论：建议由运行时层接手### 5.6 结论：建议由运行时层接手

| 我们已完成 | 建议移交 |
|---|---|
| 定位到具体 kernel（名字、调用次数、耗时）| 分析 kernel 内部为什么慢 |
| 算出每个算子占差距的比例 | 决定优化哪个（成本/收益）|
| 排除非瓶颈（GEMM/attention/layer_norm 在这些场景不是问题）| 实现融合、向量化、精度路径修正 |
| 给出 FlagGems 的实测结论（**两平台都更慢，不是解法**）| 提供原生 bf16 kernel |
| 单卡口径的算子耗时 | **多卡下通信与计算的时间分解**（§5.5）|

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

### 6.3 昇腾 910C：⚠️ 只有微基准，**排名待整步测量确认**

**这一节的数字是候选信号，不是"占差距的权重"。** 昇腾目前只跑过
`profile_op_micro_npu.py` 的微基准（`op-micro/ascend_910c.json`），
真实 step 的 profiler 还没接。

#### 逐算子实测（ms/次，bf16，B=8 S=512 H=1024 FFN=4096 vocab=151669）

| 算子 | 昇腾 910C | A100 | 昇腾/A100 |
|---|---|---|---|
| `mm_lm_head` (4096×1024 @ 1024×151669) | **7.901** | 15.62 | **0.51×（快 2×）** |
| `mm_ffn` (4096×1024 @ 1024×8192) | **0.266** | 0.352 | **0.76×（更快）** |
| `mm_hidden` (4096×1024 @ 1024×1024) | **0.076** | 0.092 | **0.83×（更快）** |
| `flash_attn_fwd` (B8 S512 GQA) | 1.819 | 1.440 | 1.26× |
| **`rms_norm`** | **0.156** | 0.024 | **6.50×** |
| **`layer_norm`** | **0.095** | 0.034 | **2.79×** |
| `sum_last` | 0.055 | 0.019 | 2.91× |
| `bmm_scores` (B8 128×128) | 0.098 | 0.033 | 2.98× |
| `softmax` | 0.073 | 0.026 | 2.81× |
| `cross_entropy` | 0.275 | 0.085 | 3.23× |
| `add` / `mul` | 0.045 / 0.044 | 0.017 / 0.017 | 2.66× / 2.59× |
| `gelu` / `silu` | 0.053 / 0.047 | 0.021 / 0.021 | 2.52× / 2.22× |

#### 与另两家的对比（**口径不同，不能混排**）

| 平台 | 口径 | 第一名 | 结论 |
|---|---|---|---|
| **昆仑芯** | 整步 profiler（占差距 %）| GELU **65.8%** | 非 GEMM 受限 |
| **PPU** | 整步 profiler（占差距 %）| GEMM **44.3%** | GEMM 受限 |
| **昇腾** | **微基准（倍数）** | RMSNorm **6.50×** | ⚠️ **也指向非 GEMM 受限，但权重未定** |

**昇腾的 GEMM 比 A100 还快（0.51-0.83×）**，这一点和 PPU 正好相反、
和昆仑芯同方向。所以"非 GEMM 受限"这个定性判断是可信的。

但**具体是 RMSNorm 还是别的算子占大头，现在说不出来**——微基准的倍数
需要乘以真实 step 的调用次数才是权重，而调用次数取决于模型结构和实测
序列长度（BGE padded 均值 299）。

#### 要补什么

```bash
# 在 NPU 上跑整步 profiler，口径与另三家一致
python examples/retrieval/profile_step_segments.py \
    --conf examples/retrieval/conf/train/bge_m3.yaml \
    --steps 3 --skip 2 --prof-steps 4 --device npu \
    --out segments_ascend.json
```

**风险**：NPU 的 profiler 可用性未验证。昆仑芯当初也是调了两个环境变量
（`XPU_ENABLE_PROFILER_TRACING=1` + `XPU_CUPTI_ENABLE_DEVICE=0`）才能用，
昇腾大概率也需要类似的前置条件，可能要找厂商确认。

#### 拿到整步数据后能回答的问题

| 现在能说 | 现在不能说 |
|---|---|
| 昇腾的 GEMM 强于 A100（0.51-0.83×）| RMSNorm 在真实 step 里占差距多少 % |
| 归一化/激活/逐元素类算子慢 2.2-6.5× | 优化哪个算子收益最大 |
| 定性：非 GEMM 受限 | 与昆仑芯谁的非 GEMM 瓶颈更严重 |

#### 建议给谁

| 项 | 建议 |
|---|---|
| **`rms_norm` 6.50×** | 运行时层——单项最差，且 BGE/reranker 都用它 |
| `layer_norm` 2.79× | 运行时层 |
| 激活（gelu/silu 2.2-2.5×）+ 逐元素（add/mul 2.6×）| 运行时层，可能靠融合解决 |
| `sum_last` 2.91× / `softmax` 2.81× / `bmm_scores` 2.98× | 先等整步权重，可能被调用次数稀释 |
| **整步 profiler 接入** | **我们补，或交运行时层** |

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
