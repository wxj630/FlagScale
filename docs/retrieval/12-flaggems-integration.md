# FlagGems 接入说明（后续算子优化的入口）

**这份文档说明：怎么在训练里打开 FlagGems、怎么验证它真的生效、
以及为什么当前基线仍然是不带的。**

## 一、一句话状态

| 项 | 状态 |
|---|---|
| 配置开关 `train.hardware.use_flaggems` | ✅ **已接通**（以前是死配置，见 §2） |
| 昆仑芯后端 | ✅ 可用（vendor `kunlunxin`，flag_gems 5.4.0） |
| PPU 后端 | ✅ 可用（vendor `thead`） |
| 默认是否开启 | ❌ **关闭**——所有吞吐数字都是不含 FlagGems 的基线 |
| 实测收益 | ❌ **昆仑芯上整体更慢**（见 §5），需针对性 exclude |

## 二、曾经的问题：`use_flaggems` 是个死配置

`examples/retrieval/conf/train.yaml` 里有 `train.hardware.use_flaggems`，
它被插值成环境变量 `FLAGOS_USE_FLAGGEMS` 交给 torchrun，
但**没有任何 Python 代码读取它**。也就是说：

```yaml
train.hardware.use_flaggems: true   # 以前：没有任何效果，静默走厂商 kernel
```

现在由 `flagscale/train/utils/flaggems.py` 的 `enable_flaggems()` 读取，
在 `train_retrieval.train()` 里、设备确定后、构建模型前调用一次。
注册是进程级的、不可撤销，所以必须在任何 forward 或被 CUDA graph 捕获的
区域之前完成。

## 三、怎么开

### 命令行

```bash
python -m flagscale.run ... \
  train.hardware.use_flaggems=true \
  train.hardware.flaggems_vendor=kunlunxin      # PPU 用 thead；留空则自动探测
```

### 配置项（`examples/retrieval/conf/train.yaml`）

| 键 | 默认 | 作用 |
|---|---|---|
| `use_flaggems` | `false` | 总开关 |
| `flaggems_vendor` | 空 | 指定后端；空=按平台自动选（`flagos`→`thead`） |
| `flaggems_exclude` | `null` | 要**跳过**的算子。`null`=用厂商默认 yaml；列表=只排除这些 |
| `flaggems_cache_dir` | 空 | JIT 产物目录；空=输出目录旁的 `flaggems_cache/` |
| `flaggems_required` | `true` | 开了但装不上时**报错**而不是静默降级 |

**`flaggems_required: true` 是刻意的**：静默回落到未优化 kernel 会让
"我开了 FlagGems"变成一句没有依据的话，而这正是 §2 那个 bug 的性质。

### 缓存目录必须改（昆仑芯上会直接失败）

FlagGems 默认把 Triton 缓存写到 `~/.flaggems`。昆仑芯上家分区是满的：

```
OSError: [Errno 28] No space left on device: '/home/jiangxuhui/.flaggems'
```

`enable_flaggems()` 会把它重定向到
`$FLAGGEMS_CACHE_DIR` / `$TRITON_CACHE_DIR` / `$TORCH_EXTENSIONS_DIR`，
优先用 `train.hardware.flaggems_cache_dir`，否则用输出目录旁的路径。
这也符合 AGENTS.md「不要在家目录工作」的约定。

## 四、怎么验证真的生效

**不要只看日志里的 `[flaggems] enabled`**——那只说明注册调用成功了。
要对比实际跑的 kernel 名：

```bash
python examples/retrieval/verify_flaggems.py \
  --conf examples/retrieval/conf/train/bge_m3.yaml --device cuda
```

它跑同一组算子两次（开/关），打印 kernel 名的差集。实测输出（昆仑芯）：

```
only with flag_gems:
  + _sum_row_full_kernel
  + _sum_row_tail_kernel
  + _tle_dsa_row_copy_kernel
  + add_func_kernel_rank_1
  + gelu_none_kernel_rank_1
  + layer_norm_oneshot_kernel
  + mm_kernel
only without:
  - xblas_xpu3::fc_one_loop<...>          (GEMM)
  - xpukernel_xpu3::gelu_highprecision_nv<float>
  - xpukernel_xpu3::layer_norm_fwd<float, float>
  - xpukernel_xpu3::reduce_mt<0, float, float>
  - xpukernel_xpu3::tensor_calc_basic<...>
```

注意**不能靠 kernel 名里有没有 "triton" 来判断**：昆仑芯上 FlagGems 的
Triton kernel 是提前编译的，报出来是普通 kernel 符号。

## 五、实测：接上了，但昆仑芯上整体更慢

单算子微基准（`examples/retrieval/profile_op_micro.py` 同形状对比，
`with flag_gems.use_gems()` 生效后测）：

| 算子 | 昆仑芯 原生 → FlagGems | 倍数 |
|---|---|---|
| `sum` | 0.025 → 0.918 ms | **0.03×** |
| `layer_norm` | 0.044 → 1.06 ms | **0.04×** |
| `add` | 0.033 → 0.183 ms | 0.18× |
| `silu_and_mul` | — | 0.27× |
| `mul` | 0.033 → 0.107 ms | 0.31× |
| `softmax` | 0.089 → 0.278 ms | 0.32× |
| `rms_norm` | 0.141 → 0.371 ms | 0.38× |
| `silu` | 0.085 → 0.218 ms | 0.39× |
| `bmm` | 0.094 → 0.181 ms | 0.52× |
| `gelu` | 0.249 → 0.244 ms | 1.02× |

**10 个里 9 个更慢。**原因是昆仑芯的 FlagGems 后端把算子拆成通用
pointwise kernel，丢掉了厂商 kernel 的融合与向量化，而厂商 kernel
本身并不差（见 `08-kunlunxin.md` 的真实 step 归因：GEMM 1.2×、
attention 0.86×，本来就不是瓶颈）。

### 三个已经定位的兼容性 bug（都会让真实训练失败）

| 崩溃点 | 报错 | 原因 |
|---|---|---|
| `torch.tensor(1.0) - mask`（transformers 构造 attention mask） | `RuntimeError: invalid argument to exchangeDevice` | FlagGems 的 `sub` 对 **CPU 标量张量**取 `in0.device.index`（为 None），直接崩 |
| `F.scaled_dot_product_attention` | `AssertionError: Currently only support dropout_p=0.0` | FlagGems 的 `_scaled_dot_product_efficient_attention` 不支持训练时的 dropout（BGE 的 attention dropout = 0.1） |
| **只开 GELU**（其余全 exclude） | `error code= 719, unspecified launch failure`（XCCL `kl3ChannelCheckErrors`） | FlagGems 的 `gelu` **反向**在 bf16 下触发 GPU kernel 异常 |

前两个可以通过 exclude 规避；**第三个绕不过去**——它恰好是昆仑芯最想优化的那个算子。

所以当前在昆仑芯上可用但不划算的配置是：

```yaml
train.hardware.use_flaggems: true
train.hardware.flaggems_vendor: kunlinxin
train.hardware.flaggems_exclude:
  - sub                        # CPU 标量 mask 路径
  - _scaled_dot_product_efficient_attention
  - _scaled_dot_product_flash_attention
  - scaled_dot_product_attention
```

### 真实 step 实测（昆仑芯，batch 10 × accum 4 = 200 texts/step）

| 配置 | 结果 |
|---|---|
| FlagGems 关闭 | **3,388 ms/step，59.0 samples/s** |
| **全量开启**（只排除 `sub` + SDPA） | ❌ **卡死**：JIT 编译 23 分钟不返回，CPU 100% |
| **只开 GELU**（其余 50+ 算子全部 exclude） | ❌ **`error code= 719`**：GPU kernel exception |

**全量开启在昆仑芯上不可用**——不是慢，是跑不完。
只开 GELU 也不行。所以"打开 FlagGems 看吞吐"这条路在昆仑芯当前版本上
走不通，必须先把这三个 bug 修掉。

### 结论：FlagGems 不是"打开就变快"的开关

它是一条**可编程的优化通道**：后续要做的不是"开 FlagGems"，
而是"针对真实 step 归因出的瓶颈算子，在 FlagGems 里写/调一个更快的实现"。
按当前归因，两家该动的算子完全不同（见 `00-index.md`）：

| 平台 | 该优化的（占差距） | FlagGems 现在能帮上？ |
|---|---|---|
| 昆仑芯 | GELU 65.8%、cast/transpose 20.6% | ⚠️ 已有 `gelu` 但持平；cast 无对应算子 |
| PPU | GEMM 44.3%、attention 20.4% | ❌ GEMM 不在范围；attention 有但被 dropout 限制 |

**所以接入的价值在于"通道打通且可验证"，不是"当前能提吞吐"。**
这一点已经量化，不要把它写成收益。

## 六、下一步

1. **昆仑芯 GELU**：厂商 `gelu_highprecision_nv` 慢 A100 25.4×、
   占总时间 35.2%，且 `approximate="tanh"` 能快 1.65×。FlagGems 的
   `gelu` 目前持平，值得针对 bf16 重写。
2. **cast/transpose 胶水**（昆仑芯，占 20.6%）：FlagGems 没有对应算子，
   需要在模型侧融合（如把 layernorm 的前后 cast 融进 kernel）。
3. **PPU 小/中 GEMM**：只能靠厂商 XBLAS，FlagGems 的 `mm` 走自己的 Triton，
   实测更慢。
4. **修 `sub` 的标量路径**：向上游报 `device.index is None` 的 bug，
   否则任何用 transformers mask 的模型都要 exclude `sub`。
5. **SDPA dropout 支持**：FlagGems 的 efficient attention 只支持
   `dropout_p=0`，训练场景基本等于不可用。

## 七、复现命令汇总

```bash
# 1. 验证开关真的改内核
python examples/retrieval/verify_flaggems.py \
    --conf examples/retrieval/conf/train/bge_m3.yaml --device cuda

# 2. 测单算子（需在装了 flag_gems 的环境里）
python examples/retrieval/profile_op_micro.py --label kunlunxin --out klx.json

# 3. 测真实 step 的吞吐（开/关各一次）
python examples/retrieval/measure_throughput.py \
    --conf examples/retrieval/conf/train/bge_m3.yaml --steps 80 --skip 20 --device cuda
```

昆仑芯运行环境额外需要（都是踩过的坑）：

```bash
export FLAGGEMS_CACHE_DIR=/data2/jiangxuhui/flaggems_cache   # 家目录已满
export XPU_ENABLE_PROFILER_TRACING=1                          # 用 profiler 时
export XPU_CUPTI_ENABLE_DEVICE=0                              # 逻辑设备号 0
```
