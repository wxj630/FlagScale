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

## 三、环境要求（2026-10-29 实测）

真机实测版本，不是猜测值：

```text
CANN            8.5.0
python          3.11.14（uv 装的 standalone 构建）
torch           2.6.0+cpu        ← 注意是 cpu wheel，配 torch_npu 用
torch_npu       2.6.0.post5
transformers    4.57.x
pyarrow         21.0.0           ← 必须钉住
decorator                        ← CANN 的 GE 编译器设 HF32 时要
```

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

> ⚠️ 两个版本坑（都实测踩过）：
> 1. `torch` 与 `torch_npu` **必须严格配套**（这里是 2.6.0 / 2.6.0.post5），
>    装错会在 `import torch_npu` 时报符号缺失。
> 2. **`pyarrow` 必须 21.0.0**。pyarrow 26 要求 NumPy 2，而 FlagScale 钉的是
>    NumPy 1.26.4，装了会冲突。
>
> 一键建环境：`bash examples/retrieval/ascend/setup-ascend-env.sh`（幂等，可重跑）。

> ⚠️ 这套 torch 2.6 比 A100（2.11）和 PPU（2.10）**低两个大版本**。检索代码用的是
> 标准 API，目前跑通了 bge / clip / reranker 的 FSDP2 路径，但 `transformers` 只能用
> 4.57 这一档，不能用 A100 上的 5.16。

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
source examples/retrieval/ascend/ascend-env.sh          # CANN + venv + FS_PLATFORM=npu

# bge_m3 / clip：一卡一个
bash examples/retrieval/run-ascend.sh bge_m3 0
bash examples/retrieval/run-ascend.sh clip_vit_large 1

# reranker：4 卡 FSDP2（见下面的显存原因）
bash examples/retrieval/run-ascend.sh qwen3_reranker 0 "0,1,2,3"
```

脚本内部等价于：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
python -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=qwen3_reranker \
  train.parallel=fsdp2 train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=4 \
  train.hardware.device_type=npu train.hardware.distributed_backend=hccl \
  train.hardware.autocast_device=npu train.hardware.use_flaggems=false \
  train.batch_size=16 train.gradient_accumulation_steps=1 \
  train.output_dir=<OUT> experiment.exp_dir=<OUT>/logs
```

**超参原则上不许写在命令行上**：batch / lr / epochs / max_length / split 全部
来自 `conf/train/<model>.yaml`（AGENTS.md 第八节）。上面只出现两处例外，
都是**昇腾实测逼出来的**，已登记在 AGENTS.md 的平台例外表：

| 项 | 共享配置 | 昇腾实测 | 原因 |
|---|---|---|---|
| 卡数 | 2 | **4** | 昇腾芯片是 **64 GB**，比 A100(80)/PPU(96) 小 |
| `batch_size` | 32 | **16**（GA 1） | 4×64 GB 下 batch 32/rank 放不下；**有效 batch 仍是 16×4=64 行/步**，与其他平台一致 |

### 5.1 关键修复：FSDP2 必须先在 CPU 上建模型再搬分片

没有这条，4 卡必 OOM。**每个 rank 都会把完整的 fp32 8B（约 33 GB）先加载进 HBM，
然后才分片** —— 4 张 64 GB 卡在第一次 all-gather 就全挂。

修复在 `train_retrieval.py` 的 `parallel_mode == "fsdp2"` 分支：

```python
# 全模型建在 CPU（host RAM 足够），fully_shard 对 CPU module + NPU mesh 是合法的
model_device = torch.device("cpu") if parallel_mode == "fsdp2" else device
model, processor = build_retrieval_model(model_type, load_cfg, model_device)
...
model = parallel_utils.apply_fsdp2(model, device_type=device.type, ...)
# 分片后才把「本 rank 的那一片」搬到 NPU，并回收空闲显存
model.to(device=device, dtype=master_dtype)
if device.type == "npu":
    torch.npu.empty_cache()
```

日志会出现 `NPU memory after FSDP2 move: allocated=… reserved=…`，用来确认只搬了分片。

### 5.2 两个显存相关的环境变量 / 参数

```bash
# FSDP2 的 all-gather 缓冲很多，关掉碎片化（昇腾支持 expandable_segments，
# 这点和昆仑芯 P800 相反）
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# bge 一卡跑 batch 10 × GA 4 时，用梯度检查点压到 18 GB HBM
# （run-ascend-bge-gc.sh 已内置 +train.model.gradient_checkpointing=true）
```

### 5.3 现成的辅助脚本（`examples/retrieval/ascend/`）

| 脚本 | 用途 |
|---|---|
| `ascend-env.sh` | `source` 用：CANN + venv + `FS_PLATFORM=npu` |
| `setup-ascend-env.sh` | 幂等建 venv（torch 2.6.0+cpu / torch_npu 2.6.0.post5 / pyarrow 21.0.0） |
| `run-ascend-bge-gc.sh` | 单卡 BGE + 梯度检查点（batch 10 / GA 4） |
| `run-ascend-reranker-4card-bs16.sh` | 4 卡 FSDP2 reranker（batch 16 × 4 卡 = 64 行/步） |
| `eval-ascend-mixed-when-done.sh` | 等 `training_complete.json` 出现后跑全量测试集评测（带 `--save-predictions`） |
| `bench_gemm_npu.py` / `bench_hf32_npu.py` / `test_fp64_npu.py` | 硬件能力探测（FP32/FP16/BF16、HF32、FP64） |

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

## 九、验证状态（2026-10-29 真机更新）

**已跑通**（昇腾真机，4×64 GB，CANN 8.5.0 / torch 2.6.0+cpu / torch_npu 2.6.0.post5）：

| 项 | 状态 | 备注 |
|---|---|---|
| `device_type=npu` 平台识别 | ✅ | `platform_npu.py` 直接可用 |
| HCCL 集合通信 | ✅ | 4 卡 FSDP2 正常 |
| coalesced all-gather shim（HCCL） | ✅ | 无需额外修改，`_process_group_classes` 的探测路径在 torch_npu 上生效 |
| bge_m3 单卡 + 梯度检查点 | ✅ | 18 GB HBM，batch 10 × GA 4 |
| clip 单卡 | ✅ | 已出 base/trained 评测 |
| reranker 8B 4 卡 FSDP2 | ✅ | 需 5.1 的 CPU-init 修复；batch 16 × 4 卡 |
| 混合精度 fp32 master + bf16 前向 | ✅ | 数值可用 |

**仍未验证**：

1. **2 卡能否跑上 reranker** —— 64 GB × 2 = 128 GB，而优化器状态就要 131 GB，
   `batch_size` 还得降到很小，实际很可能放不下。默认已改 4 卡。
2. **昇腾与其他平台的指标可比性** —— 配方有 2 处平台例外（4 卡、batch 16），
   数值上是否严格可比需要实测确认。
3. **`transformers 5.x`** —— 当前 torch 2.6 只能用 4.57 档，跨平台对比要注明。
4. **完整 3 epoch 长跑** —— 只验证了链路能跑，未验证收敛行为。
   注意：A100 与昆仑芯的 reranker 长跑在 3 epoch 内都发散了（见
   `Notes/reranker-multiplatform-run-2026-10-26.md`），昇腾大概率同样，**不要
   直接上 3 epoch**。

## 十、代码回传记录（2026-10-29）

昇腾机器只能经堡垒机手动拷贝，所以代码改动靠**打包含并**回本地，不是 `sync.sh`。

回传包内容与处理：

| 包内文件 | 处理 |
|---|---|
| `ascend-4card-cpuinit-20261029.patch` | ✅ 已 `git apply`（基线 blob `3831fa6c` 匹配，`--check` 通过） |
| `FlagScale-files/train_retrieval.py` | ✅ 作为校验基准：应用后与本地逐字节一致（仅注释措辞不同） |
| `scripts/*` | ✅ 收入 `examples/retrieval/ascend/`，并改为从脚本位置推导路径 |
| `retrieval-range-a51d3b78..8cf2e4ec.patch` | ⏭️ 跳过：本地已在 `8cf2e4ec`，无需重复应用 |
| `FlagScale-20261027.diff` / `.status` | ⏭️ 跳过：**全是权限位变化**（100755→100644），零内容变更 |
| `Torch-FL-20261027.diff` / `.status` | ⏭️ 同上，零内容变更 |
| `pre-reupload-20261027-2130.tar.gz` | 归档在 `backups/`，未合入 |

> **教训**：`scp`/`rsync -a` 传输后会丢掉可执行位，`git status` 显示几十个
> `M`，但 `git diff` 里只有 `old mode`/`new mode`。判断这类差异要看
> `grep -E "^[+-][^+-]"` 是否为空，别按文件个数下结论。
