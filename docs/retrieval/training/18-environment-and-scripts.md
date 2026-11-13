# 训练环境与脚本手册

**目标**：换一台机器，照着这份文档能把三个模型跑起来并评测。
启动命令的完整原文（含每个踩过的坑）见 [15-launch-commands.md](15-launch-commands.md)，
本文讲**环境怎么搭、脚本怎么用**。

> ⚠️ **替代了 `examples/retrieval/ENVIRONMENTS.md`。** 那份写的是
> "PyTorch 2.10.0+cpu + Torch-FL CUDA boxing"，但 A100 实际跑的是
> **torch 2.11.0+cu128 原生 CUDA、根本没装 torch_fl**。已过时。

## 一、四个平台的环境实况

| 平台 | Python | torch | transformers | runtime |
|---|---|---|---|---|
| **A100** | 3.12.13 | **2.11.0+cu128** | 5.16.1 | 原生 CUDA（无 torch_fl）|
| **昆仑芯 P800** | 3.10.22 | **2.9.0+cu128** | 4.57.6 | 厂商 XPytorch（`device_type=cuda`）|
| **PPU ZW810E** | 3.12.3 | **2.10.0** | 4.57.0 | Torch-FL `flagos`（容器内）|
| **昇腾 910C** | 待补 | 待补 | 待补 | torch_npu |

**同一份代码跑在三个 torch 版本上**（2.9.0 / 2.10.0 / 2.11.0+cu128），
这是刻意的：每家的厂商栈绑死自己的 torch。**不要试图统一成一个环境。**

顺带一个版本差异的实例：`transformers` 从 4.x 到 5.x 时 `get_text_features`
的返回类型变了，评测脚本里因此有跨版本兼容代码（`_clip_features`）。
换机器时 transformers 版本对不上，首先怀疑这类 API 漂移。

## 二、各平台的 Python 入口

| 平台 | Python | 怎么来 |
|---|---|---|
| A100 | `/flagos-search-codes/sentence-transformers/.venv/bin/python` | 预先建好的 venv，需 `export FLAGSCALE_PYTHON_ENV=<那个目录>` |
| 昆仑芯 | `/data2/jiangxuhui/envs/retrieval-vendor/bin/python` | `source .../activate` |
| PPU | `/usr/local/bin/python` | **容器内**，宿主 python 是 3.6.8 且无 torch |
| 昇腾 | 待补 | — |

### 为什么必须显式设 `FLAGSCALE_PYTHON_ENV`

FlagScale 的 runner 会在生成的脚本里 `source $FLAGSCALE_PYTHON_ENV/bin/activate`。
不设的话它用默认值 `/flagos-search-codes/torch-fl-cuda/.venv`——**那是 A100 的路径**，
在 PPU 容器里不存在，`source` 失败后整个脚本继续往下跑但环境没激活。

## 三、目录与同步

```
本机 /Volumes/PSSD/code/FlagOS/           ← 唯一代码来源
  ├── FlagScale/   ├── Torch-FL/   └── megatron-reranker/

A100        /flagos-search-codes/{FlagScale,Torch-FL,megatron-reranker}
昆仑芯      /data2/jiangxuhui/{FlagScale,Torch-FL,megatron-reranker}
PPU         /bmcp_lvm_fs/jiangxuhui/{...}（容器内 /workspace）
```

```bash
./sync.sh push                                    # 默认 A100
PLATFORM=kunlunxin ./sync.sh push                 # 指定平台
PLATFORM=pingtouge ./sync.sh check                # 双向差异检查
```

> **改完必须 push**，否则服务器跑的还是旧代码。`sync.sh` 从 `FlagScale/.env`
> 读地址，换机器只改 `.env`。细节见 AGENTS.md §三/§七。

## 四、数据与模型的路径约定

配置里只写统一路径，各机器做软链：

| 统一路径 | A100 | 昆仑芯 | PPU |
|---|---|---|---|
| `/flagos-retrieval-models` | `/flagos-search-models` | `/data2/jiangxuhui/models` | `/workspace/models` |
| `/flagos-retrieval-datasets` | `/flagos-search-datasets` | `/data2/jiangxuhui/data` | `/workspace/data` |
| `/flagos-retrieval-ckpts` | `/flagos-search-ckpts` | 同左 | `/workspace/checkpoints` |

**新容器必须重建软链**（`setup-ppu-container.sh` 里有完整命令）：

```bash
ln -sfn /workspace/models  /flagos-retrieval-models
ln -sfn /workspace/data    /flagos-retrieval-datasets
mkdir -p /flagos-retrieval-ckpts && ln -sfn /workspace/checkpoints /flagos-retrieval-ckpts/store
for m in bge-m3 clip-vit-large-patch14 Qwen3-Reranker-8B; do
  ln -sfn /workspace/models/$m /workspace/models/$m/$m; done
```

漏了会报 `HFValidationError: Repo id must be in the form 'repo_name'`。

## 五、训练脚本

### 5.1 入口

```bash
python -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=<model> \
  train.parallel=<ddp|fsdp2|megatron> \
  experiment.runner.nproc_per_node=<N> \
  train.hardware.device_type=<cuda|flagos> \
  train.output_dir=<...> experiment.exp_dir=<...>
```

**命令行只传这七类**（AGENTS.md §八）：`train=` / `action=` / `train.parallel=` /
`nproc_per_node` / `train.hardware.*` / `output_dir` / `exp_dir`。
**batch、lr、epoch、max_length、数据路径一个字都不要写**——它们来自配置，
抄到命令行就会分叉。

### 5.2 现成脚本

| 脚本 | 用途 |
|---|---|
| `run-a100.sh <model> <card> [cards]` | A100 单模型。reranker 自动切 fsdp2 双卡 |
| `run-kunlunxin.sh <model> <cards>` | 昆仑芯。reranker 走 megatron |
| `run-ppu-three.sh <model>` | PPU 容器内 |
| `run-ascend.sh <model>` | 昇腾 |
| `train-reranker-resume-a100.sh` | A100 从 checkpoint 续训（见 §5.4）|
| `watch-reranker-three.sh` | 三平台进度一条命令看完 |

### 5.3 三个模型的并行方案（差异很大，别混）

| 模型 | A100 | 昆仑芯 | PPU | 昇腾 |
|---|---|---|---|---|
| BGE-M3 | ddp 1 卡 | ddp 1 卡 | ddp 1 卡 | ddp |
| CLIP | ddp 1 卡 | ddp 1 卡 | ddp 1 卡 | ddp |
| **Reranker 8B** | **fsdp2 2 卡** | **Megatron PP=4** | fsdp2 2 卡 | fsdp2 4 卡 |

**为什么各平台不同**（细节见 [`MULTI-GPU-SCHEME.md`](../../../../MULTI-GPU-SCHEME.md)）：

- A100：单卡装不下 8B 的 fp32 优化器状态（8.19B × 16 字节 = 131 GB）
- 昆仑芯：FSDP2 需要 2.31 GiB 连续 all-gather 缓冲，厂商 shim 无
  `expandable_segments` → 只能 Megatron；**PP=2 装不下整步，PP=4 才行**
- 昇腾：单卡 64 GB，4 卡 × batch 16 才好放

### 5.4 续训的真实含义

`save_model` **只存权重**，checkpoint 里没有优化器状态、LR schedule 位置、RNG。
所以"续训"是**权重热启动**，AdamW 动量从零开始。

`train-reranker-resume-a100.sh` 的处理：把 lr 设成中断时 cosine 应该在的位置
（epoch 1 结束 = 7.86e-6，日志实测 7.860e-06 吻合），跑剩余 epoch。

## 六、评测脚本

### 6.1 `evaluate_retrieval.py`

单进程，**不需要 torchrun**。8B reranker 也是单卡评的
（PPU 96 GB 上实测全量 16,740 query，14.55 对/s）——
**单卡装不下的是训练，不是评测**：训练要 fp32 优化器状态 131 GB，评测只需 batch 4 的前向。

```bash
python examples/retrieval/evaluate_retrieval.py \
  --model-type bge_m3 --task embedding \
  --model-path /flagos-retrieval-models/bge-m3/bge-m3 \
  --data-path /flagos-retrieval-datasets/t2ranking/t2ranking/triplet-15 \
  --split test --num-negatives 15 --batch-size 32 \
  --device cuda --max-length 512 \
  --output metrics.json
```

| 参数 | 说明 |
|---|---|
| `--model-type` | `bge_m3` / `clip_vit_large` / `qwen3_reranker` / ... |
| `--task` | `embedding` / `reranker` / `clip` / `vl_embedding` |
| `--checkpoint` | 训练后的 checkpoint 目录（不给则用 `--model-path` 的原始权重）|
| `--split` | `train` / `validation` / `test`；CLIP 用 `all` |
| `--num-negatives` | 候选池大小 − 1；我们评测用 15（16 候选）|
| `--max-samples` | 只跑前 N 条，冒烟测试用 |
| `--save-predictions` | 存逐条打分（200-900 MB），之后可用 `tools/recompute_metrics.py` 重算指标 |
| `--device` | `cuda` / `flagos` / `cpu` |

**输出**：metrics JSON，含 `recall@k` / `ndcg@k` / `mrr` / `mean_rank`，
以及 `inference` 段（`items_scored` / `model_seconds` / `samples_per_s`）。

### 6.2 指标重算

```bash
python tools/recompute_metrics.py preds.json --cutoffs 1 5 10 20 --ndcg
```

从 `*.predictions.json` 重算，不用重跑模型。用于统一口径（原评测按固定 cutoff，
重算用完整候选 + nDCG 增益）。

### 6.3 `evaluate_models.sh`

三模型批量评测的封装，见脚本头部注释。

## 七、分析脚本

| 脚本 | 用途 |
|---|---|
| `tools/recompute_metrics.py` | 从 predictions 重算指标 |
| `tools/audit_retrieval.py` | 审计产物（checkpoint 能否加载、json 在不在）|
| `examples/retrieval/profile_step_segments.py` | 真实 step 分阶段计时 + profiler |
| `examples/retrieval/measure_throughput.py` | 端到端吞吐（多区间、去首尾 5%）|
| `examples/retrieval/estimate_gpu_hours.py` | GPU-hours 估算（`06-gpu-hours.md` 由它生成）|
| `examples/retrieval/diagnose_reranker_*.py` | reranker 打分分布/百分位诊断 |
| `examples/retrieval/repair_fsdp2_checkpoint.py` | 离线修复损坏的 FSDP2 checkpoint |
| `examples/retrieval/probe_card.py` | 逐卡探活（昆仑芯必做，见 AGENTS.md）|

## 八、换机器上手的检查清单

1. `./sync.sh check` —— 确认代码无差异
2. 探卡：昆仑芯必须逐卡跑 `torch.mm`（卡 1 硬件坏）
3. 确认 `FLAGSCALE_PYTHON_ENV` 指向本机真实 python
4. 确认 `/flagos-retrieval-*` 三个软链存在
5. 冒烟：`--max-samples 2000` 跑 20 步，看 loss 是否合理下降
6. 正式跑：输出目录**换新的**，别覆盖历史 checkpoint
