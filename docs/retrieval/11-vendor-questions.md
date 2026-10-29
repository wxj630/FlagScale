# 11 · 给厂商的问题清单（昆仑芯 / 平头哥）

**目的**：把应用层已经排除的方向和量化后的剩余瓶颈整理成可以直接下发给
厂商的问题。所有数字都来自**真实训练 step 的 profiler**，不是微基准。

复现条件统一为：BGE-M3（568M），batch 4 × seq 512，bf16，一次 forward+backward。
A100 同条件基线 **25.25 ms**。

| 平台 | 实测 | vs A100 |
|---|---|---|
| 昆仑芯 P800 | **175.14 ms** | 6.94× |
| PPU 810E | **110.76 ms** | 4.39× |

---

## 一、昆仑芯 P800

### 1.1 `gelu` 走了 highprecision（fp32）路径 —— 占 22%

| kernel | 每步调用 | ms | 占 step |
|---|---|---|---|
| `xpukernel_xpu3::gelu_bwd_highprecision_nv<bfloat16>` | 120 | **24.12** | 13.8% |
| `xpukernel_xpu3::gelu_highprecision_nv<bfloat16>` | 120 | **14.38** | 8.2% |

**问题**：输入是 bf16，但 kernel 名带 `highprecision`，怀疑内部按 fp32 累加。
A100 上同类算子（`gelu` + `gelu_backward`）合计只有 2.17 ms，**昆仑芯慢 17.8×**。

**想问**：
1. 是否有 bf16 精度的 gelu 前向/反向实现可选？
2. 如果有，通过什么开关启用（环境变量 / op 属性 / 编译选项）？
3. 如果 highprecision 是唯一实现，能否给出它的设计理由（比如 bf16 下精度不达标）？

> 已自查：这台机器上没有控制精度的环境变量（只有 `XBLAS_JITC_CACHE_PATH`
> 这类 JIT 缓存位置和卡选择变量）。

### 1.2 胶水 kernel 占 33%，A100 上不存在这些

| kernel | 每步调用 | ms | 占 step | 怀疑用途 |
|---|---|---|---|---|
| `hydra_internal_kernel::cast_te(bf16 const*, float...)` | **3,610** | **21.03** | 12.0% | bf16↔fp32 转换 |
| `xpu3::findmax1d_cluster<bf16>` | **3,610** | **15.27** | 8.7% | 归约（与 cast 次数完全一致） |
| `xpukernel_xpu3::tensor_calc_basic<0, bf16, float...>` | 2,286 | **13.58** | 7.8% | 通用逐元素 |
| `xpukernel_xpu3::transpose_0213_cluster<float16>` | 2,160 | **8.18** | 4.7% | 布局转换 |

**问题**：`cast_te` 与 `findmax1d_cluster` 都是 **3,610 次/步**，次数完全一致，
怀疑是某个算子内部固定做了“转 fp32 → 求 max → 转回”的两步操作。
`tensor_calc_basic` 是通用兜底 kernel（名字叫 basic，2,286 次），
说明有不少 elementwise 运算没有专用融合实现。
`transpose_0213` 是 0-2-1-3 转置，说明有 layout 不匹配需要搬运。

**想问**：
1. `cast_te` + `findmax1d_cluster` 这对是否来自某个具体算子？能否融合成一个 kernel？
2. `tensor_calc_basic` 覆盖了哪些 op？哪些有专用 kernel 可用但没被路由到？
3. `transpose_0213` 的输入输出 layout 是什么？能否通过调整前一个算子的输出 layout 消除？
4. 厂商是否有 **kernel 融合** 的工具或图优化开关（类似 `torch.compile` 的厂商后端）？

**为什么重要**：这四项合计 58.06 ms，占 33.2%。A100 上这些要么被融合掉、
要么成本可忽略。这是**融合不足**的结构性问题，不是单个算子慢。

### 1.3 小 GEMM（`fc_bf16_rn_one_loop`）

| kernel | 每步调用 | ms | 占 step |
|---|---|---|---|
| `xblas_xpu3::fc_bf16_rn_one_loop<ArgsJL1_1565...>` | 720 | 13.42 | 7.7% |
| `xblas_xpu3::fc_bf16_rn_one_loop<ArgsJL1_1749...>` | 720 | 12.84 | 7.3% |
| `xblas_xpu3::fc_bf16_rn_one_loop<ArgsJL1_1659...>` | 725 | 12.84 | 7.3% |
| `xpu3::tensor_calc_k_nk` | 725 | 7.42 | 4.2% |
| `xpu3::reduce_tn` | 720 | 5.84 | 3.3% |

**问题**：三种实例合计 39.10 ms（22%）。A100 上对应 6.73 ms，慢 **5.8×**。
注意这**不是**大 GEMM 的问题——同一模型在 LM head（151669×4096）那种大 GEMM 上
昆仑芯反而比 A100 快 15%。

**想问**：batch 4、seq 512、hidden 1024 这类**小 M 形状**的 fp16/bf16 GEMM，
XBLAS 是否有特化路径？是否需要设置 `XBLAS_JITC_CACHE_PATH` 之外的调优参数？

### 1.4 已确认不是问题的项（不用厂商处理）

| 项 | 实测 | 说明 |
|---|---|---|
| `flash_attention` | 6.30 ms（1.26×） | 实现良好，占 1% |
| `layer_norm` | 8.17 ms（1.75×） | 可接受 |
| 大 GEMM | 比 A100 快 15% | — |

### 1.5 昆仑芯复现脚本

```bash
source /data2/jiangxuhui/flagos-retrieval-vendor-env.sh
export CUDA_VISIBLE_DEVICES=2 XPU_VISIBLE_DEVICES=2 \
       XPU_ENABLE_PROFILER_TRACING=1 XPU_CUPTI_ENABLE_DEVICE=0 \
       PYTHONPATH=/data2/jiangxuhui/FlagScale
python /data2/jiangxuhui/step_profile.py \
  --model /flagos-retrieval-models/bge-m3/bge-m3 --model-type bge_m3 \
  --task embedding --max-length 512 --batch 4 --out /tmp/klx.json
```

> 注意 profiler 的两个坑：必须**同时**设 `XPU_ENABLE_PROFILER_TRACING=1` 与
> `XPU_CUPTI_ENABLE_DEVICE=0`（**逻辑设备号 0**，不是物理卡号）；
> 且 event key 是裸 kernel 名（不是 `aten::`），过滤时不能按前缀写。

---

## 二、平头哥 PPU 810E

### 2.1 小 GEMM 占 56% —— 头号问题

| op | 每步调用 | ms | 占 step |
|---|---|---|---|
| `aten::mm` | **1,444** | **43.73** | **39.5%** |
| `aten::addmm` | 725 | **18.65** | **16.8%** |
| `aten::bmm` | 720 | 7.81 | 7.1% |
| **合计** | 2,889 | **70.19** | **63.4%** |

对应底层 kernel 是 `gemm_ktype0_aiu1_mtype1_dtypeBF16xBF16xFP32xBF16xBF16...`
（多种实例，合计约 62 ms）。

**问题**：A100 上同形状只要 6.73 ms，**慢 10.4×**（含 bmm 时；只算 mm+addmm 是 9.3×）。这是 BGE-M3 encoder 的
小矩阵乘（batch 4，seq 512 → [1024, 460] × [460, 1024] 这类形状）。

**想问**：
1. `gemm_ktype0_aiu1_mtype1` 是不是通用兜底路径？有没有针对小 M 的专用 kernel？
2. 是否支持类似 cuBLAS 的 `CUBLAS_WORKSPACE_CONFIG` 或算法选择开关？
3. `mm_ppu_expand.yaml`（FlagGems 里 661 行的 matmul 调优表，含 `mm_ppu_small_m`
   等 18 个形状组）**只作用于 FlagGems 自己的 Triton kernel**，对原生
   `aten::mm` 无效。厂商是否有等价的 XBLAS 侧调优入口？

### 2.2 归约与拷贝

| op | 每步调用 | ms | 占 step |
|---|---|---|---|
| `aten::sum` | 725 | 10.21 | 9.2% |
| `aten::bmm` | 720 | 7.81 | 7.1% |
| `aten::add_` | 1,758 | 7.93 | 7.2% |
| `aten::_to_copy` | 1,120 | 4.54 | 4.1% |

**想问**：`sum` 725 次是否来自同一个算子（如 layernorm 内部的 mean）？
`bmm` 720 次（BGE 的 batch 内负例打分）能否有更大 BLOCK 的专用实现？
`_to_copy` 1,120 次是什么 dtype/layout 转换？

### 2.3 已确认不是问题的项

| 项 | 实测 | 说明 |
|---|---|---|
| `gelu` | 2.01 ms，**0.93×（比 A100 快）** | 无需处理 |
| `layer_norm` | 3.27 ms，**0.70×（比 A100 快）** | 无需处理 |

### 2.4 PPU 复现脚本

```bash
docker exec flagos-retrieval-bge bash -c "
  cd /workspace/FlagScale &&
  FLAGSCALE_DEVICE=flagos FS_PLATFORM=flagos GEMS_VENDOR=thead \
  PYTHONPATH=/workspace/FlagScale \
  python /tmp/ppu_step_profile.py \
    --model /flagos-retrieval-models/bge-m3/bge-m3 --model-type bge_m3 \
    --task embedding --max-length 512 --batch 4 --out /tmp/ppu.json"
```

> 注意：PPU profiler 会**重复计数**——同时输出 `aten::` 层和底层 kernel 层，
> 两者相加正好等于报告的总时间。取 `aten::` 层（110.76 ms）即可。

---

## 三、两家的短板完全不同（不要用同一套优化方案）

| | 昆仑芯 P800 | PPU 810E |
|---|---|---|
| 头号瓶颈 | `gelu` highprecision（22%）+ 胶水 kernel（33%） | **小 GEMM（63%）** |
| `gelu` | 17.77× 慢 | **0.93×（更快）** |
| `layer_norm` | 1.75× 慢 | **0.70×（更快）** |
| 大 GEMM | **比 A100 快 15%** | 未单测 |
| 主要手段 | kernel 融合 + 激活精度 | GEMM 库调优 |

## 四、已排除的方向（不用再试）

| 方向 | 结论 | 依据 |
|---|---|---|
| FlagGems 替换 elementwise | ❌ 昆仑芯上 10 个算子里 9 个**更慢**（`sum` 37×、`layer_norm` 26×） | `08-kunlunxin.md` |
| FlagGems 的 `mm_ppu_expand.yaml` 调优 | ❌ 只作用于 FlagGems 自己的 Triton kernel，不碰原生 `aten::mm` | 源码 + 实测 |
| 逐算子微基准排序优化目标 | ❌ 会给出**相反**的结论（如 flash attention 微基准 2.04×、真实 step 1.26×） | `08` 的方法论段落 |
| 昆仑芯 `device_type=cuda` 走 vendor 路线 | ✅ 已在用（这是当前方案） | — |

## 五、附：为什么必须用真实 step 归因

同一批算子，两种测法结论相反：

| 项 | 微基准 | 真实 step |
|---|---|---|
| `flash_attention` | 2.04× 慢，判为头号 | **1.26×，只占 1%** |
| `mm` | 0.85×（比 A100 快） | **5.81×（慢）** |
| `gelu` | 11.9× | **17.77×** |

原因：微基准每次调用都同步，且形状与真实 step 不同（真实 step 的 `mm` 是
[1024,460]×[460,1024]，微基准用的是 1024×1024）；真实 step 里的调用是
流水线化的，单次耗时低于同步测量。
