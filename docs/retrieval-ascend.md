# 昇腾（Ascend NPU）上跑 retrieval：手拷运行手册

> 平台：**昇腾 NPU + torch_npu + HCCL**
> 代码：本机 `FlagScale/` 同步过去的那一份（AGENTS.md 第一节）
> 机器：只能经堡垒机 + Mac terminal 登录，**agent 无法直连**，所以本文按
> 「本机准备好 → 手拷 → 手动跑」的顺序写。
> 状态：**代码已就绪，尚未在真机验证**。下面标 ⚠️ 的地方都是没跑过的。

## 一、为什么昇腾比 PPU 简单

| | PPU | 昇腾 |
|---|---|---|
| 设备层 | Torch-FL PrivateUse1（`flagos` 设备） | **torch_npu 原生 `npu` 设备** |
| 分布式后端 | FlagCX / NCCL | **HCCL** |
| 需要 `torch_fl` 吗 | **需要** | **不需要** |
| FlagScale 平台 | `flagos` | `npu`（`platform_npu.py`） |

`flagscale/platforms/platform_npu.py` 就是几个 `torch.npu.*` 调用（`is_available`
/ `device` / `device_count` / `dist_backend()="hccl"` / `amp_device_type()="npu"`），
和 `platform_cuda.py` 同一层级。所以**不引入 Torch-FL 这层依赖**。

## 二、本机准备

代码改动已在 `FlagScale` 里提交，直接同步过去即可：

```bash
# 本机
cd /Volumes/PSSD/code/FlagOS
PLATFORM=ascend ./sync.sh push        # ⚠️ sync.sh 里还没有 ascend，见下
```

`sync.sh` 目前**没有 `PLATFORM=ascend`**。两条路：

1. **手拷（推荐，只需一次）**：
   ```bash
   # 本机
   rsync -az --no-owner --no-group \
     --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' \
     --exclude 'build/' --exclude '*.so' --exclude 'logs/' \
     /Volumes/PSSD/code/FlagOS/FlagScale/  <user>@<npu-host>:~/FlagOS/FlagScale/
   ```
2. 给 `sync.sh` 加一条 `ascend)` 分支（需要远端路径），之后就能用 `check` 做双向校验。

## 三、环境要求

```bash
# CANN
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Python 环境里需要：
python 3.10/3.11
torch           2.1~2.4（与所装 torch_npu 版本严格匹配）
torch_npu       ← 关键，device 名是 npu
transformers    4.57.x
sentence-transformers
omegaconf hydra-core numpy pyarrow pandas pillow safetensors wandb
```

> ⚠️ `torch` 与 `torch_npu` 版本必须严格配套，装错会在 `import torch_npu` 时
> 报符号缺失。

## 四、路径（和别的平台同一套约定）

配置里只写 `/flagos-retrieval-*`，在昇腾机器上做软链：

```bash
sudo mkdir -p /flagos-retrieval-models /flagos-retrieval-datasets /flagos-retrieval-ckpts
sudo ln -sfn <你的 models 目录>   /flagos-retrieval-models
sudo ln -sfn <你的 datasets 目录> /flagos-retrieval-datasets
sudo ln -sfn <你的 ckpts 目录>    /flagos-retrieval-ckpts
```

数据目录结构要和训练配置一致：

```
<datasets>/t2ranking/t2ranking/triplet-15/*.parquet   # 11 个文件, 340,111 行
```

## 五、跑训练

```bash
cd ~/FlagOS/FlagScale
export FLAGSCALE_PYTHON_ENV=<装了 torch_npu 的环境>

# bge_m3 / clip：一卡一个
bash examples/retrieval/run-ascend.sh bge_m3 0
bash examples/retrieval/run-ascend.sh clip_vit_large 1

# reranker：两卡 FSDP2（131 GB 优化器状态，单卡放不下）
bash examples/retrieval/run-ascend.sh qwen3_reranker 0 "0,1"
```

脚本内部等价于：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
python -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=qwen3_reranker \
  train.parallel=fsdp2 train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=2 \
  train.hardware.device_type=npu train.hardware.distributed_backend=hccl \
  train.hardware.autocast_device=npu train.hardware.use_flaggems=false \
  train.output_dir=<OUT> experiment.exp_dir=<OUT>/logs
```

**超参一个都不要写在命令行上**：batch / lr / epochs / max_length / split 全部
来自 `conf/train/<model>.yaml`（AGENTS.md 第八节）。

## 六、这次为昇腾改了哪些代码

| 文件 | 改动 | 为什么 |
|---|---|---|
| `flagscale/train/utils/parallel.py` | `install_coalesced_allgather_shim` 从只补 `ProcessGroupNCCL` 改成遍历所有可导入的后端类（NCCL / **HCCL** / Gloo） | 昇腾上根本没有 `ProcessGroupNCCL`，原来的实现会直接 `return False`，FSDP2 起不来 |
| 同上 | 把内嵌的 fallback 提到模块级 `_allgather_into_tensor_coalesced` | 多个类共用一个实现 |
| `flagscale/train/train_retrieval.py` | `init_distributed` 的设备校验：`elif requested not in {...}` 改成显式 `cuda` 分支 + 通用 `else` 分支 | 原来 `npu` 会走进「unavailable」分支报错 |
| 同上 | DDP `device_ids`：`platform.name() in {"cuda","npu","musa"}` | torch_npu 不接受 `device_ids=None` |
| 同上 | `seed_everything` 里 cuDNN 开关加 `hasattr` 保护 | 昇腾没有 cuDNN |
| 同上 | `reproducibility_record` 的 platform 兜底改成 `device.type` | 原来硬编码 `"cuda"`，会把昇腾 run 标错 |
| `examples/retrieval/run-ascend.sh` | 新增 | 启动脚本 |

## 七、首次运行必须盯的几件事

按顺序验证，每步失败都能定位到具体层：

### 1. 设备能起来

```bash
python -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```

### 2. 平台被识别

```bash
FS_PLATFORM=npu python -c "
from flagscale.platforms import get_platform
p = get_platform(); print(p.name(), p.dist_backend(), p.device(0))"
```

应当打印 `npu hccl npu:0`。

### 3. 单卡 10 步 smoke（bge，最便宜）

```bash
cd ~/FlagOS/FlagScale
python -m flagscale.run --config-path examples/retrieval/conf --config-name train \
  action=test train=bge_m3 train.parallel=ddp \
  experiment.runner.nproc_per_node=1 \
  train.hardware.device_type=npu train.hardware.distributed_backend=hccl \
  train.hardware.autocast_device=npu train.hardware.use_flaggems=false \
  train.output_dir=/tmp/smoke-bge experiment.exp_dir=/tmp/smoke-bge/logs
```

盯：loss 是否下降、`lr=` 是否在 warmup 后正常衰减、有没有算子 fallback 到 CPU 的警告。

### 4. 双卡 HCCL（reranker 前先验集合通信）

```bash
python -c "
import torch, torch_npu, torch.distributed as dist
dist.init_process_group('hccl', init_method='env://')
t = torch.ones(4, device='npu')
dist.all_reduce(t); print(t)"
```
用 `torchrun --nproc_per_node=2` 跑这段。

### 5. reranker 2 卡 FSDP2

```bash
bash examples/retrieval/run-ascend.sh qwen3_reranker 0 "0,1"
```

⚠️ 最可能卡住的两处：

- **coalesced all-gather shim 是否生效**：日志里应出现
  `coalesced all-gather shim installed on ProcessGroupHCCL`。没出现说明
  `torch_npu._C._distributed_c10d.ProcessGroupHCCL` 的导入路径和这里假设的不同，
  需要按实际类名改 `_process_group_classes()`。
- **单卡显存**：8.19B × 16 字节 = 131 GB，两卡各 66 GB。昇腾 910B 单卡 64 GB
  **偏紧**，可能需要 3-4 卡，或参考昆仑芯的先例换并行方案。

## 八、训练完必须评测

```bash
cd ~/FlagOS/FlagScale
MODELS="bge_m3 clip_vit_large" \
RUN_ROOT=/flagos-retrieval-ckpts/ascend-fp32master-3epoch \
EVAL_DEVICE=npu \
PYTHON=$FLAGSCALE_PYTHON_ENV/bin/python \
NUM_NEGATIVES=15 \
bash examples/retrieval/evaluate_models.sh
```

reranker 记得 `BATCH_SIZE=8`（8B + seq 640 余量小），且 `NUM_NEGATIVES=15`
（训练是 1，评测要用数据集的 16 候选全集）。

## 九、还没验证的（当成未知，别当结论）

1. **coalesced all-gather 在 HCCL 上的类名/行为** —— 代码按 `torch_npu` 的公开
   路径写，真机可能需要调整。
2. **FSDP2 在 HCCL 上是否完整可用** —— `init_device_mesh("npu", ...)` 理论上可行，
   未实测。
3. **bge_m3 的动态 shape** —— 昆仑芯上踩过「每步重编译」的坑；昇腾若有 graph
   编译缓存，需要确认 `max_length` 固定后不再重编译。
4. **混合精度**：`master_dtype=fp32` + bf16 前向在 910B 上是否与 CUDA 数值一致。
5. **torch_npu 版本的 autocast 设备串**：代码传 `autocast_device=npu`，
   若该版本不支持会回退成无 autocast（`autocast_context` 里已 try/except 兜底）。
