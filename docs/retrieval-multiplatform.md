# Retrieval 模型各平台运行手册（BGE-M3 / CLIP / Reranker-8B）

本文记录 **BGE-M3**、**CLIP ViT-L/14** 和 **Qwen3-Reranker-8B** 在四台机器上
**实际跑通的命令**与**环境构成**。每个数字都对应一次真实运行，不是设计稿。

> 配置只有一份：`examples/retrieval/conf/`。换机器只改**路径和设备**，
> 超参（batch / lr / epochs / max_length / 数据 split）一律不动，否则结果不可比。
> 详见仓库根目录 `AGENTS.md` 第八节。

**昇腾（Ascend）不在这份文档里**：它走 `torch_npu` 原生 `npu` 设备 + HCCL，
不需要 Torch-FL，而本机无法直连那台机器（只能堡垒机手拷）。单独看
[`retrieval-ascend.md`](retrieval-ascend.md)。

**reranker 与 bge/clip 的本质区别**：8B 全参训练需要 16 字节/参数
（fp32 主权重 + 两个 fp32 Adam 矩 = 8.19B × 16 ≈ **131 GB**），单张 80/96 GB 卡
放不下。所以 bge/clip 是「一卡一个模型」的 DDP，**reranker 必须两卡并行**，
而且不同平台用的并行实现并不相同：

| 平台 | reranker 并行策略 | 为什么 |
|---|---|---|
| A100 | **FSDP2** | 标准 torch 支持，分片到 2 卡 |
| PPU | **FSDP2** | 已跑通；需补一个 all-gather shim |
| 昆仑芯 P800 | **Megatron PP=2 / TP=1** | FSDP2 在 P800 上只能 bf16 master |

两种策略的取舍见第五节。

## 一、总览

| 平台 | 运行时 | 设备类型 | bge/clip 并行 | reranker 并行 | 环境形态 |
|---|---|---|---|---|---|
| A100（bitahub_2gpus） | 原生 CUDA（标准 torch） | `cuda` | DDP（1 卡/模型） | **FSDP2（2 卡）** | **venv** |
| PPU（pingtouge） | Torch-FL `flagos` 设备 | `flagos` | DDP（1 卡/模型） | **FSDP2（2 卡）** | **镜像 + 容器** |
| 昆仑芯 P800（kunlunxin） | 厂商 XPytorch（`torch_xmlir`） | `cuda` | DDP（1 卡/模型） | **Megatron PP=2（2 卡）** | **venv** |
| A100（bitahub，旧 1 卡） | Torch-FL CUDA boxing | `flagos` | DDP | 不适用 | venv（历史基线） |

**三种运行时的区别**：

- **原生 CUDA**：标准 PyPI torch，`torch.cuda` 就是真 CUDA。没有 Torch-FL。
- **Torch-FL `flagos`**：`import torch_fl` 抢注 PrivateUse1，张量在 `flagos` 设备上，
  算子由 Torch-FL 分发（PPU 走 CUDA boxing，全量算子）。
- **厂商 XPytorch**：厂商的 torch 发行版把 XPU 伪装成 `torch.cuda`，
  所以代码里写 `cuda` 就行，**但 venv 里没有 `torch_fl`**（`ModuleNotFoundError`）。

## 二、A100（bitahub_2gpus，2×A100-80GB）

### 环境：venv（关键：没有 torch_fl）

```text
venv:  /flagos-search-codes/sentence-transformers/.venv
python 3.12.13（home = /opt/conda/bin，用 uv 创建）
torch          2.11.0+cu128      ← 标准 PyPI CUDA 构建
transformers   5.16.1
hydra-core     1.3.2
numpy          1.26.4
```

**这台机器没有 `torch_fl`**，走的是纯原生 CUDA。之前 A100 的 fp32-master 基线
（`train-a100-fp32master.sh`）用的就是同一个 venv，所以历史结果可比。

> ⚠️ 两个 venv 陷阱：
> 1. `/flagos-search-codes/torch-fl-cuda/.venv` 是**坏的**（缺 `bin/python`），
>    FlagScale 配置里 `before_start` 默认 source 的正是它 → 报
>    `torch-fl-cuda/.venv/bin/torchrun: cannot execute`。
>    必须显式 `export FLAGSCALE_PYTHON_ENV=<好的 venv>`。
> 2. 这台 transformers 是 **5.16.1**，其他平台是 4.57。tokenizer 行为可能有差异，
>    跨平台对比时要注明。

### 路径

配置里写 `/flagos-retrieval-{models,datasets,ckpts}`，这台机器需要软链（已建）：

```bash
ln -sfn /flagos-search-models    /flagos-retrieval-models
ln -sfn /flagos-search-datasets  /flagos-retrieval-datasets
ln -sfn /flagos-search-ckpts     /flagos-retrieval-ckpts
```

### 运行命令

一张卡跑一个模型（两张卡并行）：

```bash
# 卡 0: BGE-M3
bash examples/retrieval/run-a100.sh bge_m3 0    # 在 FlagScale 根目录执行

# 卡 1: CLIP
bash examples/retrieval/run-a100.sh clip_vit_large 1
```

脚本内部（`examples/retrieval/run-a100.sh`）：

```bash
export CUDA_VISIBLE_DEVICES=<card>
export FLAGSCALE_PYTHON_ENV=/flagos-search-codes/sentence-transformers/.venv
export PATH="$FLAGSCALE_PYTHON_ENV/bin:$PATH"
export PYTHONPATH=/flagos-search-codes/FlagScale

python -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=<model> \
  train.parallel=ddp train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=1 \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  train.hardware.autocast_device=auto train.hardware.use_flaggems=false \
  train.output_dir=/flagos-retrieval-ckpts/a100-fp32master-3epoch/<model> \
  experiment.exp_dir=<same>/logs
```

### reranker：两张卡一起跑 FSDP2

```bash
# 占据 0,1 两张卡；parallen 自动切到 fsdp2、nproc 2
bash examples/retrieval/run-a100.sh qwen3_reranker 0 "0,1"
```

`run-a100.sh` 对 `qwen3_reranker` 自动做三件事（其余模型仍是 DDP 单卡）：

1. `train.parallel=fsdp2`、`experiment.runner.nproc_per_node=2`；
2. 第三个参数是卡列表，写成 `CUDA_VISIBLE_DEVICES="0,1"`（torchrun 在重映射后的
   命名空间里从 0 开始编号，所以必须给连续集合）；
3. 输出仍在 `/flagos-retrieval-ckpts/a100-fp32master-3epoch/qwen3_reranker`，
   和 `evaluate_models.sh` 的 `RUN_ROOT/<model>` 约定一致。

FSDP2 把参数/梯度/优化器状态都分片到两张卡，所以 131 GB 的优化器状态只需
每卡约 66 GB。**不要**把 bge/clip 也改成 FSDP2 —— 它们是单卡 DDP 跑出来的
基线，换并行方式会改变数值，历史结果就不可比了。

## 三、PPU（pingtouge，ZW810E，单卡 96 GiB）

### 环境：镜像 + 容器

```text
镜像:  flagtree-ppu-py312-torch2.10.0-sdk2.1.0-cu130-ubuntu24.04:202607-3.6-base
容器:  flagos-retrieval（挂载 /bmcp_lvm_fs/jiangxuhui → /workspace，--network=host）
OS:    Ubuntu 24.04.2 LTS
python 3.12.3
torch          2.10.0            ← 标准 torch，厂商 CUDA 兼容层在 SDK 里
torch_fl       2.10.0+ppu2.0.0   ← editable 装到 /workspace/Torch-FL
transformers   4.57.0
hydra-core     1.3.7
numpy          1.26.0
```

厂商栈不是 torch 发行版，而是**SDK 层的 CUDA 兼容运行时**
（`/usr/local/PPU_SDK/CUDA_SDK/lib64/libcuda.so`、`libcudart.so.13`、`libnccl.so`），
所以 `torch.cuda` 直接可用、`device_name` 报 `PPU-ZW810E`。

### 两个必须设的环境变量

```bash
export GEMS_VENDOR=thead          # PPU 是平头哥(the head)；不设则 FlagGems 探测失败退出
export NCCL_SOCKET_IFNAME=bond0   # 不设则 NCCL 报 "no socket interface found"
```

### 容器内路径（`/` 只读，软链必须建在容器内）

```bash
ln -sfn /workspace/models /flagos-retrieval-models
ln -sfn /workspace/data   /flagos-retrieval-datasets
mkdir -p /flagos-retrieval-ckpts && ln -sfn /workspace/checkpoints /flagos-retrieval-ckpts/store
# 配置用两级布局 <name>/<name>，PPU 数据是单层，补内层软链：
for m in bge-m3 clip-vit-large-patch14 Qwen3-Reranker-8B; do
  ln -sfn /workspace/models/$m /workspace/models/$m/$m
done
for d in t2ranking wds_mscoco_captions2017; do
  ln -sfn /workspace/data/$d /workspace/data/$d/$d
done
```

一键完成：`bash examples/retrieval/setup-ppu-container.sh`（容器内，仓库挂在 /workspace）

### 运行命令

一个模型一个容器、一张卡（互不干扰）：

```bash
# BGE-M3（容器 flagos-retrieval-bge，绑 ppu6）
docker exec -d flagos-retrieval-bge bash -c \
  "nohup bash /workspace/FlagScale/examples/retrieval/run-ppu-three.sh bge_m3 > /workspace/logs/launch-bge.log 2>&1"

# CLIP（容器 flagos-retrieval-clip，绑 ppu7）
docker exec -d flagos-retrieval-clip bash -c \
  "nohup bash /workspace/FlagScale/examples/retrieval/run-ppu-three.sh clip_vit_large > /workspace/logs/launch-clip.log 2>&1"
```

脚本内部（`examples/retrieval/run-ppu-three.sh`）关键部分：

```bash
export FLAGSCALE_DEVICE=flagos FS_PLATFORM=flagos
export FLAGSCALE_DISTRIBUTED_BACKEND=flagos FLAGSCALE_AUTOCAST_DEVICE=flagos
export GEMS_VENDOR=thead NCCL_SOCKET_IFNAME=bond0 PYTHONPATH=/workspace/FlagScale
export TMPDIR=/flagos-retrieval-ckpts/.../tmp          # 容器 / 小，别用 /tmp

python -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=<model> \
  train.parallel=ddp train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=1 \
  train.hardware.device_type=flagos train.hardware.distributed_backend=flagos \
  train.hardware.autocast_device=flagos train.hardware.use_flaggems=false \
  train.output_dir=<ckpt>/<model> experiment.exp_dir=<ckpt>/<model>/logs
```

### reranker：两张卡一起跑 FSDP2

容器 `flagos-retrieval` 绑 `ppu12,ppu13`，一个容器同时看到两张卡：

```bash
docker exec -d flagos-retrieval bash -c \
  "cd /workspace/FlagScale && nohup bash examples/retrieval/run-ppu-three.sh qwen3_reranker \
   > /workspace/logs/launch-reranker.log 2>&1"
```

脚本里 reranker 分支与 bge/clip 的差别只有并行方式和卡数：

```bash
train.parallel=fsdp2 train.master_dtype=fp32 \
experiment.runner.nproc_per_node=2
```

实测显存 **94.9 / 93.9 GB（96 GB 卡的 97%）**，几乎没有余量。因此：

- `--eval-batch-rows` / eval 的 `BATCH_SIZE` 必须压到 **4-8**，否则 eval 阶段 OOM；
- 存盘走 `cpu_offload=True` 的全量 state dict（`parallel_utils.full_state_dict`），
  不会在卡上再分配 32 GB，所以 epoch 末存盘是安全的；
- 别和其他模型共用这两张卡。

吞吐实测约 **5.3 对/s**（128 对 / 24 s）。

### reranker 的 FSDP2 额外需要 shim

PPU 厂商 `ProcessGroupNCCL` 缺 `allgather_into_tensor_coalesced`，FSDP2 会直接
`AttributeError`。FlagScale 在 `flagscale/train/utils/parallel.py` 里装了替代实现
（逐个 `_allgather_base`），`apply_fsdp2()` 自动调用，无需额外操作。

这个 shim 走了四轮才成功，值得记下为什么：

1. 给 `ProcessGroupNCCL` 加类属性 —— 无效，FSDP2 调的是 C++ op，不看 Python 属性；
2. `torch.library` 注册 IMPL —— 无效，`_c10d_functional` 的 CUDA dispatch key
   在 `Functional.cpp` 里静态注册过；
3. 改 `funcol.all_gather_tensor` 但传 group 名 —— `dist.get_backend("0")` 失败；
4. **最终**：在 `funcol.all_gather_tensor` 里直接调 `pg._allgather_base`，
   绕过带日志的 wrapper —— 成功。

**这条修改在 FlagScale 侧，没有动 Torch-FL**（AGENTS.md 第六节）。

## 四、昆仑芯 P800（kunlunxin，XPU）

### 环境：venv（厂商 XPytorch，无 torch_fl）

```text
venv:  /data2/jiangxuhui/envs/retrieval-vendor
python 3.10
torch          2.9.0+cu128       ← 厂商 torch_xmlir（XPU 伪装成 cuda）
transformers   4.57.6
hydra-core     1.3.7
numpy          2.2.6
torch_fl       ✗ 没有
```

环境变量必须 source：

```bash
source /data2/jiangxuhui/flagos-retrieval-vendor-env.sh
```

它设置：`FLAGSCALE_DEVICE=cuda`、`FS_PLATFORM=cuda`、
`LD_LIBRARY_PATH`（含 `torch_xmlir/xhpc/xblas/dependency_so`、`/usr/local/xpu/so`）、
`TMPDIR=/data2/jiangxuhui/tmp`。

### 运行命令

```bash
# BGE-M3 在卡 5，CLIP 在卡 4（一张卡一个模型）
bash examples/retrieval/run-kunlunxin.sh bge_m3 5    # 在 FlagScale 根目录执行
bash examples/retrieval/run-kunlunxin.sh clip_vit_large 4
```

脚本内部（`examples/retrieval/run-kunlunxin.sh`）：

```bash
source /data2/jiangxuhui/flagos-retrieval-vendor-env.sh
export CUDA_VISIBLE_DEVICES=<card> XPU_VISIBLE_DEVICES=<card>
export TMPDIR=/data2/jiangxuhui/tmp      # 宿主 / 只剩 ~120MB
export FLAGOS_USE_FLAGGEMS=false

$PY -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=<model> \
  train.parallel=ddp train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=1 \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  train.hardware.autocast_device=auto train.hardware.use_flaggems=false \
  train.output_dir=/data2/jiangxuhui/flagos-retrieval-ckpts/<model> \
  experiment.exp_dir=<same>/logs
```

> ⚠️ **卡 1 硬件损坏**（任何多 GB 拷贝触发 `wait for noc idle timeout`，status=299）。
> 卡 2/3/6/7 长期被其他租户的 vLLM 占用且会在训练中途增长。选卡前先看
> `xpu-smi`，并预留 20 GB 以上余量。

### reranker：两张卡跑 Megatron PP=2（**不用 FSDP2**）

昆仑芯的 reranker **不走 FlagScale**，而是 `megatron-reranker/`（Megatron-LM-FL）。

**FSDP2 在 P800 上卡住的原因，不是"lm_head 没被切分"**——切分是生效的
（日志里 `oversized leaf: backbone.lm_head gathers 1185 MiB` 指的是
**all-gather 之后的完整缓冲**，不是未分片副本）。真正的机制是：

- FSDP2 每个 param group 在前向 pre-hook / 反向 pre-backward 要把整组参数
  all-gather 成**一块连续缓冲**，每个 rank 都临时持有一份完整权重；
- `lm_head` 是 151669×4096，fp32 下这块缓冲 = **2.31 GiB**，而且**不随卡数缩小**
  （缩小的只是分片，缓冲始终是完整尺寸），所以加卡无用；
- P800 不支持 `expandable_segments`（VMM 的 `cuMemAddressReserve` 返回
  `801 NOT_SUPPORTED`），分配器必须找到**连续物理段**。实测 28.43 GiB 空闲仍申请
  不出 2.31 GiB —— 碎片化了；
- 把 `embed_tokens` / `lm_head` 各自拆成独立 group 后，最大单次请求从 2.31 GiB
  降到 **1.16 GiB**（bf16），这样 bf16 master 能跑；
- 但 fp32 master 下这块缓冲仍是 2.31 GiB，**所以 P800 的 FSDP2 只能停在
  bf16 master（8 字节/参数）**，拿不到 A100 基线的 16 字节布局。

Megatron 绕开的是这个缓冲本身：PP 把层切到不同 stage，每张卡只持有自己那些层，
前向不需要把整个模型的参数 all-gather 起来；distributed optimizer 再分片
fp32 主权重 + 两个 fp32 矩，于是拿到 A100 基线用的 **16 字节/参数**。

```bash
cd /data2/jiangxuhui/megatron-reranker
CARDS=2,3 PP=2 NPROC=2 bash run_8b.sh      # 全部超参默认值来自配置语义
```

`run_8b.sh` 默认值（已与 `conf/train/qwen3_reranker.yaml` 对齐）：

| 变量 | 默认 | 对齐配置的哪一项 |
|---|---|---|
| `PP` / `NPROC` | 3 / 3（脚本默认）；正式跑用 **2 / 2** | 2 卡时 PP=2（18/18 层）。TP 固定 1——32 个 head 不能按 3 切 |
| `NUM_MICRO` | 4 | 需设为 **64** 才等于配置的 `batch_size: 32` × 2 卡 |
| `MAX_LENGTH` | 640 | `model.max_length` |
| `EPOCHS` | 1 | 需设为 **3** |
| `CACHE_DIR` | `work/token-cache-split` | 按 query 划分的缓存（必须有 `splits` 字段） |
| `TRAIN_ROWS` | 空 = 全量 | `data.max_samples: null` |
| `LR` / `MIN_LR` | **1e-7 / 1e-8** | ⚠️ 见下 |

**正式跑法的完整命令**：

```bash
CARDS=2,3 PP=2 NPROC=2 NUM_MICRO=64 MAX_LENGTH=640 EPOCHS=3 \
CACHE_DIR=/data2/jiangxuhui/megatron-reranker/work/token-cache-split TRAIN_ROWS= \
OUT=/data2/jiangxuhui/megatron-reranker/runs/reranker8b-pp2-split \
EVAL_EVERY=250 SAVE_EVERY=1000 EVAL_BATCH_ROWS=4 \
EXTRA="--warmup-steps 200 --skip-nonfinite" \
setsid bash run_8b.sh > logs/pp2-split-outer.log 2>&1 < /dev/null &
```

> `setsid` 不是可选项：SSH 里 `nohup ... &` 起的进程会随会话退出被杀，
> 表现为「重启后日志没变、进程表为空」，很容易误判成启动成功。

#### lr 必须改成 1e-7（P800 的平台例外）

共享配置是 `lr 1e-5 → 1e-6 cosine`。在 P800 上用 1e-5 实测：

| step | 训练端 | eval |
|---|---|---|
| 138 | loss 0.0127 / acc 1.0000 / grad_norm 3.46 | — |
| 198 | loss **0.0000** / acc 1.0000 / grad_norm **1.000**（正好等于 clip 阈值） | — |
| 250 | loss 0.0377 / acc 0.9844 | **loss 13.7928 / acc 0.5000** |

eval acc 0.5 = 完全随机。换成 1e-7 后同一步：**eval loss 0.3771 / acc 0.8568**。

`Notes/megatron-reranker-p800.md` §11 的扫描结论一致：1e-6 在 200 步内发散、
3e-7 先降后回涨、**1e-7 是唯一能长跑的值**。这条例外已登记在 `AGENTS.md` §八，
并且是 `run_8b.sh` 的默认值。

#### 训练完必须导出才能评测

Megatron 存的是每个 stage 一个 `stage{K}.pt`，评测器读不了。用一条命令
「导出 → 评测」：

```bash
bash /data2/jiangxuhui/FlagScale/examples/retrieval/finish-kunlunxin-reranker.sh
```

它做两件事：

1. `python -m mrr.export_hf --stages <run>/checkpoint --pp 2 --out <HF 目录>`
   把 stage 合并回标准 HF checkpoint；
2. 软链到 `<EVAL_ROOT>/qwen3_reranker` 后调用 `evaluate_models.sh`，
   base + trained 各评一遍，16 候选，写 predictions。

**注意**：`mrr/train.py` 存盘前会先 `flush_optimizer_to_model()`。不做这一步
`model.state_dict()` 是**过时的 bf16 副本**，曾经出现「评测健康、存盘权重 1e34」。

## 五、reranker 的两种并行策略：FSDP2 vs Megatron

bge/clip 是 0.6B 级，一卡一个模型用 DDP 就够了。reranker 8.19B 全参微调的
优化器状态（fp32 主权重 + 两个 fp32 矩）就有 131 GB，**必须切分**。两条路：

| | FSDP2 | Megatron PP |
|---|---|---|
| 切什么 | 参数/梯度/优化器状态按**张量**分片 | 按**层**把模型切成若干 stage |
| 每卡放什么 | 每一层的一小片 | 连续的若干整层 |
| 通信 | all-gather 参数（前向/反向各一次） | stage 间只传激活（p2p） |
| 额外算子要求 | `allgather_into_tensor_coalesced` | 无（用 PP 自带的 send/recv） |
| 框架 | FlagScale `train.parallel=fsdp2` | Megatron-LM-FL `mrr/train.py` |
| 已验证平台 | A100、PPU | 昆仑芯 P800 |

### 为什么不是同一个方案通吃

- **昆仑芯不能用 FSDP2 跑 fp32 master**：FSDP2 的切分本身是生效的，卡住的是
  **all-gather 后的完整缓冲**——每个 param group 在每个 rank 上都要拼出一块完整
  参数，`lm_head`（151669×4096）在 fp32 下就是 2.31 GiB，且不随卡数缩小。
  P800 没有 `expandable_segments`，只能找连续物理段，28 GiB 空闲也失败。
  bf16 下这块降到 1.16 GiB（再把它拆成独立 group），所以 FSDP2 只能停在
  **bf16 master**，丢掉 16 字节/参数的 fp32 主权重。
  Megatron 的 PP 不做整体 all-gather，distributed optimizer 又能分片 fp32 状态，
  才拿到 16N。
- **PPU 不用 Megatron**：Megatron-LM-FL 在 PPU 上没验证过，而 FSDP2 只需补一个
  all-gather shim 就能跑（见第三节）。按「适配开发量最小」原则选 FSDP2。
- **A100 两者都行**，选 FSDP2 因为它是 FlagScale 原生路径，和 bge/clip 同一套
  启动/存盘/评测链路。

### 怎么选（按顺序判断）

1. 平台原生 torch 的 FSDP2 能跑 → **用 FSDP2**（链路最短，评测脚本直接可用）。
2. FSDP2 因分配器/算子缺失退化精度 → **用 Megatron PP**，并接受多一个
   `export_hf` 步骤。
3. 两者都不行 → 先确认是不是卡的问题（P800 的卡 1 是坏的），别急着换框架。

### 两条路的对齐要求

两者必须产出**可比的数字**，所以要盯这三项：

| 项 | 要求 | 怎么验 |
|---|---|---|
| 数据划分 | 同一套 query 分桶 | 两边日志都应打印 `train=306806 validation=16565 test=16740` |
| 全局 batch | 64 行/步（= 128 对/步） | FlagScale `batch_size 32 × 2 卡`；Megatron `NUM_MICRO 64` |
| lr 调度 | cosine + warmup 200 | 两边日志都打印实际 lr，逐段核对 |

> Megatron 侧之前犯过一次：`NUM_MICRO` 用默认 4，就是 8 行/步，只有配置的 1/8。
> 这类偏差不会报错，只会让结果不可比 —— 所以启动后必须回去看 `plan:` 那行。

## 六、性能与结果对照

### bge/clip（历史基线，bf16 时代，**已作废待重测**）

| 模型 | 平台 | 训练时长 | 备注 |
|---|---|---|---|
| BGE-M3 | A100 1 卡 | 22.8 h | 3 epoch |
| BGE-M3 | PPU 1 卡 | 18.5 h | 3 epoch，比 A100 快 1.23× |
| CLIP | A100 1 卡 | 1.4 h | 3 epoch |
| CLIP | PPU 1 卡 | 1.3 h | 3 epoch，比 A100 快 1.08× |

> 这些是**纯 bf16** 时代的数字。当前跑的是 `master_dtype=fp32` 的混合精度实验，
> 且数据切分已改为按 query 分桶（旧的按行切分让 89.4% 测试 query 泄漏进训练），
> 所以**旧指标全部作废**，需要重测。

### reranker 8B（fp32 master，2 卡，实测）

| 平台 | 并行 | 吞吐 | 步速 | 14,379 步预计 | 单卡显存 |
|---|---|---|---|---|---|
| 昆仑芯 P800 | Megatron PP=2 | 6.3-6.5 对/s | 20 s | ~3.3 天 | 75 GB / 96 GB |
| PPU | FlagScale FSDP2 | ~5.3 对/s | 24 s | ~4.0 天 | 95 GB / 96 GB |
| A100 | FlagScale FSDP2 | 待测 | — | — | — |

步数口径：train 306,806 行 ÷ 64 行/步 = 4,793 步/epoch × 3 = **14,379 步**。
两个平台都是这个数，说明数据规模和切分确实对齐。

> PPU 的 95 GB 是 96 GB 卡的 97%，只剩约 1 GB。这个余量下 eval 批量必须 ≤8，
> 且不能和别的模型共用卡。昆仑芯的 75 GB 是 PP=2 时每 stage 18 层的结果。

## 七、跑之前先跑 audit

```bash
python tools/audit_retrieval.py --data /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15
```

它检查（不需要模型，秒级）：LR 调度是否正确、eval 是否默认全量、
配置语义字段是否齐全、query 切分是否零泄漏。这四项中任何一项错了，
都是几小时到几天的 GPU 时间白烧。

## 八、训练完必须评测

```bash
cd <FlagScale root>
MODELS="bge_m3 clip_vit_large" \
RUN_ROOT=/flagos-retrieval-ckpts/<run> \
EVAL_DIR=/flagos-retrieval-ckpts/<run>/eval \
bash examples/retrieval/evaluate_models.sh
```

base 与 trained 会自动成对测量（同一次调用、同一口径）。默认会同时写出
`<metrics>.predictions.json`（每条 query 的完整分数向量），
之后可用 `tools/recompute_metrics.py` 改 cutoff / 加 nDCG，**不用重跑模型**。

### reranker 的三个额外注意点

```bash
# FSDP2 路线（A100 / PPU）：checkpoint 已是 HF 格式，直接评
MODELS="qwen3_reranker" NUM_NEGATIVES=15 BATCH_SIZE=8 \
RUN_ROOT=/flagos-retrieval-ckpts/<run> EVAL_DEVICE=<cuda|flagos> \
bash examples/retrieval/evaluate_models.sh

# Megatron 路线（昆仑芯）：先导出 stage 权重，再评
bash examples/retrieval/finish-kunlunxin-reranker.sh
```

1. **`NUM_NEGATIVES=15`，不是训练时的 1**。训练每行只配 1 个负例
   （`conf` 里 `num_negatives: 1`），但 t2ranking 的 triplet 文件提供
   1 正 + 15 负共 **16 候选**，评测要用全部 16 个才能和 A100/PPU 对比。
2. **`BATCH_SIZE` 压到 8**。8B + seq 640 在 96 GB 卡上余量很小，默认 12 会 OOM。
3. **昆仑芯的 `--device cuda`**（不是 `flagos`）。厂商 XPytorch 把 XPU 暴露成
   `cuda`，venv 里没有 `torch_fl`；写 `flagos` 会 `ModuleNotFoundError`。
