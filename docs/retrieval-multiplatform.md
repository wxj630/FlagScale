# Retrieval 模型各平台运行手册（BGE-M3 / CLIP）

本文记录 **BGE-M3** 和 **CLIP ViT-L/14** 在四台机器上**实际跑通的命令**与**环境构成**。
每个数字都对应一次真实运行，不是设计稿。

> 配置只有一份：`examples/retrieval/conf/`。换机器只改**路径和设备**，
> 超参（batch / lr / epochs / max_length / 数据 split）一律不动，否则结果不可比。
> 详见仓库根目录 `AGENTS.md` 第八节。

## 一、总览

| 平台 | 运行时 | 设备类型 | 并行 | 环境形态 |
|---|---|---|---|---|
| A100（bitahub_2gpus） | 原生 CUDA（标准 torch） | `cuda` | DDP | **venv** |
| PPU（pingtouge） | Torch-FL `flagos` 设备 | `flagos` | DDP | **镜像 + 容器** |
| 昆仑芯 P800（kunlunxin） | 厂商 XPytorch（`torch_xmlir`） | `cuda` | DDP | **venv** |
| A100（bitahub，旧 1 卡） | Torch-FL CUDA boxing | `flagos` | DDP | venv（历史基线） |

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

### reranker 的 FSDP2 额外需要 shim

PPU 厂商 `ProcessGroupNCCL` 缺 `allgather_into_tensor_coalesced`，FSDP2 会直接
`AttributeError`。FlagScale 在 `flagscale/train/utils/parallel.py` 里装了替代实现
（逐个 `_allgather_base`），`apply_fsdp2()` 自动调用，无需额外操作。

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

## 五、性能与结果对照（历史基线，bf16 时代，**已作废待重测**）

| 模型 | 平台 | 训练时长 | 备注 |
|---|---|---|---|
| BGE-M3 | A100 1 卡 | 22.8 h | 3 epoch |
| BGE-M3 | PPU 1 卡 | 18.5 h | 3 epoch，比 A100 快 1.23× |
| CLIP | A100 1 卡 | 1.4 h | 3 epoch |
| CLIP | PPU 1 卡 | 1.3 h | 3 epoch，比 A100 快 1.08× |

> 这些是**纯 bf16** 时代的数字。当前跑的是 `master_dtype=fp32` 的混合精度实验，
> 且数据切分已改为按 query 分桶（旧的按行切分让 89.4% 测试 query 泄漏进训练），
> 所以**旧指标全部作废**，需要重测。

## 六、跑之前先跑 audit

```bash
python tools/audit_retrieval.py --data /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15
```

它检查（不需要模型，秒级）：LR 调度是否正确、eval 是否默认全量、
配置语义字段是否齐全、query 切分是否零泄漏。这四项中任何一项错了，
都是几小时到几天的 GPU 时间白烧。

## 七、训练完必须评测

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
