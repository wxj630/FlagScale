# 19 · 国产芯片适配方案与工作量说明

**给甲方的说明文档。** 目标：讲清"标准流程是什么、每款芯片为什么要单独适配、
我们改了什么、工作量在哪"。

所有内容都有代码或实测日志依据，对应提交号在文中标注。

---

## 一、标准流程（A100 基准）

A100 是**基准平台**，也是唯一不需要任何适配的平台——代码按照原生 CUDA + PyTorch
的方式写，其余平台向它对齐。

### 1.1 技术栈

| 项 | 值 |
|---|---|
| Python / torch / transformers | 3.12.13 / **2.11.0+cu128** / 5.16.1 |
| 运行时 | **原生 CUDA**，不装 torch_fl、不装任何厂商插件 |
| 设备类型 | `train.hardware.device_type=cuda` |

> 这是刻意的设计：**基准要尽可能"素"**。任何厂商插件都可能改变数值或调度行为，
> 一旦装上，其它平台与它的差异就说不清是芯片差异还是插件差异。

### 1.2 三个模型的并行方案

| 模型 | 参数量 | 并行 | 卡数 | 为什么 |
|---|---|---|---|---|
| BGE-M3 | 568M | DDP | 1 | 单卡轻松装下 |
| CLIP ViT-L/14 | 428M | DDP | 1 | 同上 |
| **Qwen3-Reranker-8B** | 8.19B | **FSDP2** | **2** | 单卡装不下 fp32 优化器的状态 |

**8B 的显存账**（这是所有适配的起点）：

```
8.19B 参数 × 16 字节/参数（fp32 权重 + 一阶矩 + 二阶矩）= 131 GB
A100 单卡 80 GB → 必须切分
```

FSDP2 把参数、梯度、优化器状态都按 rank 切分，2 卡后每卡约 66 GB，可行。

### 1.3 标准作业流程（SOP）

```
改本地代码（唯一来源）
  ↓
./sync.sh push PLATFORM   ← 推到目标机
  ↓
探卡（昆仑芯必做）
  ↓
冒烟：--max_samples 2000，跑 20 步看 loss
  ↓
正式训练（输出目录必须换新的）
  ↓
存盘验证（能否加载出真实权重）
  ↓
评测（base vs trained，同一次调用）
```

**四条不许违反的规矩**（已写入 `AGENTS.md`，踩过坑）：

1. **只在本地改代码**，服务器上只跑不改
2. **命令行只传七类参数**（模型名、并行方式、设备、卡数、输出路径、exp 路径、action），
   超参一律来自配置——抄到命令行必然分叉
3. **有训练在跑的机器不要 push**
4. **训练完必须评测**，只交 loss 不算结果

---

## 二、四个平台的适配方案

### 2.1 总览

| 平台 | 运行时路线 | 设备类型 | 主要适配点 | 代码改动 |
|---|---|---|---|---|
| **A100** | 原生 CUDA | `cuda` | 无（基准）| 0 |
| **昆仑芯 P800** | 厂商 XPytorch（CUDA 垫片） | `cuda` | bmm JIT、无 expandable_segments、FSDP2 不可用 → **Megatron PP=4** | 大 |
| **PPU ZW810E** | Torch-FL（PrivateUse1） | `flagos` | 平台注册时机、容器卡号、all-gather 垫片 | 中 |
| **昇腾 910C** | torch_npu（PrivateUse1） | `npu` | **HCCL 不支持 float64 all-reduce**、FSDP2 需 CPU 先建模型 | 小 |

**工作量与芯片"像不像 CUDA"成反比**：昆仑芯把自己伪装成 CUDA，表面最省事，
实际因为垫片不完整反而改动最大；昇腾是真正的 PrivateUse1 设备，API 最干净，
改动最小。

### 2.2 昆仑芯 P800：改动最大

昆仑芯的厂商 torch 把加速器重命名成 `cuda`，所以代码层面 `device_type=cuda` 就能跑——
**但垫片的完整度不够，这是所有问题的来源**。

#### (a) XBLAS bmm JIT 编译崩溃

**现象**：`seq >= 288` 时反向传播崩溃：

```
error: explicit instantiation of 'bmm_one_loop' does not refer to a function template
       (bmm::algo2:: vs ::bmm::algo2::)
```

**一度误判**：以为是显存限制，把 seq 降到 256 就"好了"。实际不是内存问题
（崩在 75/96 GiB），是厂商 XBLAS 的 JIT 模板名有 bug。

**解法**：`XMLIR_BMM_DISPATCH_VALUE=0` 换用另一个 bmm 内核
（`bmm_bf16_rn_one_loop`），**顺带快 2×**（0.091 ms vs 0.179 ms @seq640）。
（`b6253ab8`）

> 这一条是典型的"厂商运行时缺陷，我们只能绕过"。

#### (b) 无 `expandable_segments` → FSDP2 不可用 → 转 Megatron

**现象**：P800 的 CUDA 垫片不支持 `PYTORCH_ALLOC_CONF=expandable_segments`。
8B 的 FSDP2 需要一次性分配 **2.31 GiB 连续 all-gather 缓冲**（`lm_head`），
碎片化后分配失败，只能降级到 bf16 master 权重——**精度不达标**。

**解法**：改用 **Megatron-LM 的 pipeline parallel**。它每卡只持一部分层，
不需要整模型 all-gather，且分布式优化器仍能切分 fp32 状态。

#### (c) PP 必须 = 4，=2 不可行

这是最曲折的一段，也是工作量最大的地方：

| PP | 结果 |
|---|---|
| **PP=2** | 装不下整步 32 行 → OOM（96 GiB 只剩 1.42 GiB）→ 被迫拆成 32 个 micro-batch（每批 2 对）→ **BCE 方差极大，训练在 step 170-255 崩**（loss=0，grad_norm 撞 clip 阈值）|
| **PP=4** | 每卡 3.2 GiB 参数、**92.6 GiB 空闲** → 整步 64 对一个 micro-batch → **稳定，1,000 步零崩塌** |

**为什么 PP=2 的小 micro-batch 会崩**：每批只有 1 正 1 负，BCE 的批间方差压倒信号。
实测证据：**lr 从 1e-7 降到 1e-9（100 倍），loss 曲线几乎不变**——说明更新量相对
批间噪声可忽略。

> **中途的误判与排除**（都留了记录）：
> 先怀疑厂商 bmm JIT（包装 bmm 拿到真实报错，发现是 **OOM**）；
> 试过 `beta2=0.95→0.999`、`micro_batch_rows=4`、`clip_grad=1000`、
> fp32 模型参数，**全部经对照实验证明无效并回退**。
> 最后用 bmm/matmul 追踪器定位到真实失败是显存，才找到 PP=4 这条路。

#### (d) Megatron 路径的六个框架级缺陷

走 Megatron 之后暴露出一串我们代码自身的缺陷（不是厂商问题，是我们第一次用这条路）：

| 缺陷 | 后果 | 修复 |
|---|---|---|
| `false_token_id` 默认 9692 解码成 `'icken'`（不是 `'no'`=2152）| **BCE 卡在 3.7-5.7 → NaN，模型完全无法学习**，且不报错 | `68037120` |
| `rotary_base` 的 setattr 写在 `return` 之后（死代码）| 默认 1e4 覆盖 Qwen3 的 1e6，attention 输出 cosine 只有 0.04 | `37aa739c` |
| `_score_processor` 用 `context["rows"]` 选张量轴（只在建 batch 的 rank 上有效）| PP 两个 stage 拿到的 batch 宽度不同，崩 | `14bff37c` |
| 数据按全局 rank 分片 | **PP 两个 stage 看到不同数据**，stage1 拿别人的 hidden state 配自己的标签 | `a3a95fe1` |
| `evaluate()` 的 micro-batch 粒度与声明不符 | epoch 跑完崩在验证，**丢失 16 小时训练** | `c2b17ec4` |
| epoch 存盘写在评测**之后** | 同上，评测崩则权重全丢 | `c2b17ec4` |

> 这六个里，**第一个最致命也最难发现**：两个无关 token 的 logit 相减仍是有限数，
> 所以 loss 有值、不报错，只是永远学不到东西。

#### (e) 评测 micro-batch 与训练步长不一致

PP=4 训练时 32 行/micro-batch（64 对），`evaluate()` 拿同一个值切评测块，
但评测块只有 8 对：

```
RuntimeError: hidden batch 64 != len(lengths)=8
```

修法要**两处同时改**：切分按 chunk 大小 clamp，`micro_batch_size` 也要传 clamp 后的值
——只改前者会把不匹配移到 p2p 张量上。（`8899d2eb`）

### 2.3 PPU ZW810E：中等改动

PPU 走 Torch-FL 的 **PrivateUse1 设备**（`device_type=flagos`），是正确的设备抽象，
但引入了几类问题：

#### (a) 平台注册时机

**现象**：

```
RuntimeError: device_type=flagos was requested, but a Torch-FL platform is unavailable
```

**根因**：`register_platforms()` 在 `flagscale.platforms` 首次 import 时执行一次。
PPU 上**那一刻 Torch-FL 的设备表还是空的**（`torch.flagos.is_available()` 返回 False），
于是一个平台都没注册。

**排查代价**：这条报错信息**完全误导**——它说"Torch-FL 装错了"，实际是注册时机问题。
我们查了 Python 环境、PYTHONPATH、import 顺序、torchrun、模块遮蔽，全部排除后
才在真实进程里插桩看到 `PLATFORMS == []`。

**解法**：`init_distributed` 里在平台为空时**重新注册**，并先 `torch.flagos.device_count()`
把设备表唤醒。（`40183dda`）

#### (b) 容器只挂 2 张卡，卡号在容器内重编号

**现象**：选了"空闲"的卡 15，训练起不来，报的还是上面那条平台不可用。

**根因**：容器 `flagos-retrieval` 只映射了宿主的 **ppu12/ppu13**，容器内编号为 **0/1**。
卡号 15 在容器里根本不存在 → `device_count()=0` → 一堆连锁反应。

**教训**：`ppu-smi` 显示的是**宿主卡号**，容器内要重新编号。查法：
```bash
docker inspect <container> --format '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}'
```

#### (c) all-gather 垫片

PPU 的厂商 `ProcessGroupNCCL` **缺 `allgather_into_tensor_coalesced`**，
而 FSDP2 用它做参数组 all-gather。

**解法**：在 `flagscale/train/utils/parallel.py` 里装一个 per-tensor 回退垫片
（`install_coalesced_allgather_shim`），**不改 Torch-FL**。启动日志会打印
`coalesced all-gather shim installed on ProcessGroupNCCL`。

#### (d) 容器环境

- 容器内无 venv，`FLAGSCALE_PYTHON_ENV` 必须显式设成 `/usr/local`
  （默认值是 A100 的路径，`source` 失败后脚本继续跑但环境没激活）
- `GEMS_VENDOR=thead`、`NCCL_SOCKET_IFNAME=bond0` 缺一不可
- 新容器要重建 `/flagos-retrieval-*` 软链，否则报 `HFValidationError`

### 2.4 昇腾 910C：改动最小

昇腾用 `torch_npu`，是标准的 PrivateUse1 设备，API 干净。只遇到一个硬性阻塞：

#### (a) HCCL 不支持 float64 all-reduce

**现象**：reranker 在 epoch 1 验证结束时崩溃：

```
HCCL allreduce: Unsupported data type at::kDouble
```

**根因**：`reduce_metrics` 用 `torch.float64` 做跨 rank 求均值，HCCL 没有实现
`at::kDouble` 的 all-reduce。

**解法**：NPU 上用 `float32`，其他平台保持 `float64`——**CUDA 侧数值一字不变**，
不影响与基准的对比。（`9ec255e2`）

```python
dtype = torch.float32 if device.type == "npu" else torch.float64
```

#### (b) FSDP2 必须先在 CPU 建模型

昇腾卡 64 GB（A100 80 GB / PPU 96 GB），每个 rank 先加载完整 fp32 8B 约 33 GB
会直接 OOM。

**解法**：先在 CPU 建模型并分片，再把本地 shard 移到设备。这个改动对所有平台生效，
顺带降低了峰值显存。

#### (c) 配置例外

4 卡 × batch 16（其他平台 2 卡 × batch 32）。**有效 batch 都是 64 行/步**，
只是每卡切分不同，指标可比。

---

## 三、适配工作量汇总

### 3.1 按类别

| 类别 | 具体项 | 平台 |
|---|---|---|
| **运行时缺陷绕过** | bmm JIT 崩溃、无 expandable_segments、HCCL float64 | 昆仑芯、PPU、昇腾 |
| **框架能力补齐** | all-gather 垫片、平台注册时机 | PPU |
| **并行方案重写** | FSDP2 → Megatron PP=4 | 昆仑芯 |
| **我们代码的缺陷** | 6 个 Megatron 路径 bug + FSDP2 存盘 3 个缺陷 | 全平台 |
| **环境/部署** | 容器卡号、软链、venv、网卡 | PPU、昆仑芯 |

### 3.2 关键数字

| 项 | 值 |
|---|---|
| 代码改动 | **+2,824 行 / −25 行**，7 个文件 |
| 新增文档 | **26 篇**（平台结果、验收对照、启动手册、数据集目录等） |
| 修复的缺陷 | **13 个**（3 个 FSDP2 存盘、6 个 Megatron、4 个平台适配） |

**代码改动的分布**（近两周 `git diff --stat`）：

| 文件 | 行数 | 内容 |
|---|---|---|
| `utils/megatron_pp.py` | **+958** | 新增：Megatron pipeline 训练器（昆仑芯路线）|
| `train_retrieval.py` | +742 | 平台分支、存盘修复、评测、续训 |
| `models/retrieval/qwen3_reranker_megatron.py` | +401 | 新增：HF→Megatron 权重转换 |
| `utils/parallel.py` | +290 | all-gather 垫片、FSDP2 路径 |
| `train/arena.py` | +179 | 新增：P800 显存分配器 |
| `utils/flaggems.py` | +177 | 新增：FlagGems 接入 |
| `datasets/retrieval.py` | +102 | 按 DP rank 分片 |
| 因缺陷损失的 GPU 时间 | 约 **31 小时**（16 h + 15 h 两次 epoch 完成后崩溃） |

> **这 31 小时是可量化的适配成本**：两次都是"训练跑完、存盘/评测阶段崩溃"，
> 因为当时的代码把存盘写在评测之后。现在存盘已前置。

### 3.3 一个反直觉的结论

**最像 CUDA 的平台改动最大。** 昆仑芯把自己包装成 CUDA（`device_type=cuda` 就能跑），
但垫片不完整——bmm 编译崩、不支持 expandable_segments、FSDP2 无法工作——
最终不得不整套换到 Megatron 路径，连带暴露 6 个框架缺陷。

而昇腾是标准的 PrivateUse1 设备，只改了一行 `reduce_metrics`。

**给甲方的建议**：评估国产芯片的适配成本时，**不要看"是否兼容 CUDA"**，
要看"关键算子（bmm/attention/通信原语）的实现完整度"。

---

## 四、当前状态与遗留问题

### 4.1 三个模型的完成度

| 模型 | A100 | 昆仑芯 | PPU | 昇腾 |
|---|---|---|---|---|
| BGE-M3 | ✅ | ✅ | ✅ | ✅ |
| CLIP | ✅ | ✅ | ✅（重训复核）| ✅ |
| **Reranker 8B** | ✅ epoch 1 已评（R@1 +1.0 / R@5 −2.0）| ❌ epoch 1 已评：**R@1 退化到随机（0.063 vs 0.0625）** | ✅ 3 epoch 训完；epoch-1 trained 评测待跑 | ⏳ |

### 4.2 已知未解决

| 问题 | 影响 | 建议 |
|---|---|---|
| **昆仑芯 reranker 学不动/饱和** | lr=1e-5 饱和（val 2.914、**R@1 退化到随机**）、1e-7 学不动（bf16 分辨率）。**epoch 1 已跑完 33h，已导出并评测，确认无可部署配置** | 见 §4.3 |
| **PPU reranker 的 trained 评测** | 训练 3 epoch 已完成，base 已评（14.55 对/s 全量）；trained 需要容器内跑，agent 无 docker 权限 | 用 `run-ppu-three.sh` 在容器内补跑 |
| 昇腾环境版本未采集 | 文档缺一行 | 需在堡垒机登录后补 |
| **A100 reranker 微调净效果为负** | epoch 1：R@1 +1.02 点但 R@5 −2.04、mean_rank +0.126 | 不是芯片问题，是配方/数据问题（16 候选、单负例）|

### 4.3 昆仑芯 reranker 的深层问题（需要厂商介入）

**现象（2026-11-11 完整 epoch 1 实测，33 小时）**：lr=1e-5 下 train_loss 0.0628
但 **validation_loss 2.914**（A100 是 0.1799 / 0.5342）——train 低 2.9 倍、val 差
5.5 倍，train/val 比 **46.4×**（A100 是 3.0×）。训练后期 **54% 的步 loss=0.0000**，
68% 的步 grad_norm 卡在裁剪阈值上。这是 **BCE 饱和**，不是收敛。
lr=1e-7 时则完全不学（同一数据重复两个 epoch，loss 只降 0.003）。

**根因**：Megatron 路径的**模型参数是 bf16**（FSDP2 是 fp32）。权重典型量级 0.0114，
bf16 在该量级相邻两个可表示值是 `0.0113525390625` / `0.01141357421875`，
**ULP = 6.10e-05**。而 lr=1e-7 的单步更新只有 **0.16% 个 ULP**——需要 **610 步**
才够让一个小权重跨一格。小权重长期冻结在 bf16 网格上。

> 前一版这里写的是 ULP 2.23e-05 / 0.45% / 222 步，**算错了**（2.23e-05 是 w≈0.004
> 量级的 ULP，被套到了 0.0114 上）。结论方向不变，量级差 2.7 倍。

**`train.master_dtype=fp32` 在这条路径上是死参数。** `megatron_pp.py` 的
`OptimizerConfig` 把 `bf16=True, params_dtype=torch.bfloat16` 写死（553-556 行），
`master_dtype` 只在 FSDP2 分支被读取。所以命令行传 `master_dtype=fp32` 不报错、
也不生效——昆仑芯 pp4 run 跑满 33 小时，参数一直是 bf16。
**改这个之前先确认它真的被消费**，否则又是一次空转。

**为什么不能简单改 fp32**：改成 fp32 模型参数后，`XMLIR_BMM_DISPATCH_VALUE=0`
不再能绕过 XBLAS 的 bmm JIT 崩溃（该开关只对 bf16 路径有效）。

**已加的护栏**（`train_retrieval.py` §`_train_megatron`）：这条路现在会**拒绝**
`master_dtype` 不是 bf16 的请求并报错退出，`metrics.json` 里也显式写了
`"param_dtype": "bfloat16"`。原因是三个启动脚本原本都传 `master_dtype=fp32`，
而这条路径**静默忽略**它——33 小时的 run 用 bf16 参数训练，命令行和所有产物
却都显示 fp32。`run-kunlunxin.sh` 现在对 megatron 传 bf16、对 ddp 传 fp32。

> **这条的普适教训**：参数传了不生效比传错更危险。传错会立刻崩，
> 不生效会让整份实验记录撒谎。凡是新增一个跨路径的配置项，
> 要么在每条路径都消费它，要么在不支持时直接报错。

**目前没有可用配置**。可行的方向（都未验证）：
1. fp16 替代 bf16（ULP 细 80 倍），需验证 bmm 开关在 fp16 下有效
2. 换对 logit 幅度有约束的 loss
3. 厂商修复 XBLAS 的 bmm JIT

---

## 五、相关文档

| 文档 | 内容 |
|---|---|
| [`MULTI-GPU-SCHEME.md`](../../../../MULTI-GPU-SCHEME.md) | 运行时 / 并行策略 / 框架三层的完整矩阵 |
| [15-launch-commands](15-launch-commands.md) | 各芯片启动命令原文与踩坑记录 |
| [18-environment-and-scripts](18-environment-and-scripts.md) | 环境清单与脚本手册 |
| [20-slow-op-profiling](../analysis/20-slow-op-profiling.md) | 关键慢算子的识别方法与脚本 |
| [08-kunlunxin](../results/08-kunlunxin.md) | 昆仑芯结果与 reranker 排查全过程 |
