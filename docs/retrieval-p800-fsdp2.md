# P800 Qwen3-Reranker-8B FSDP2 验证

2026-11-13，在八张 P800 OAM（每卡 96 GiB）上验证。
本实验使用 FlagScale 原生 retrieval trainer、fp32 主权重和 AdamW 状态、bf16
计算及梯度归约。历史实验的卡故障和内存限制记录不适用于本次测试机器。

## 环境隔离与版本

代码基于 `feat/retrieval-flagos` 的
`28f6f003f92d9c86e964456e327e43c2be6866fd` 加本文对应的修改。
部署位置使用环境变量表示：`FLAGSCALE_PYTHON_ENV` 指向独立 Python 环境，
`RETRIEVAL_ROOT` 指向模型、数据和输出的工作目录。Python 环境建议放在本地盘，
避免共享文件系统上大量小文件的复制和安装开销。

```text
Python         3.10
PyTorch        2.9.0+cu128
Transformers   4.57.6
Hydra          1.3.2
NumPy          2.2.6
PyArrow        21.0.0
hyperparameter 0.5.12
XPytorch       XMLIR-p800_release_v3.7.0_torch29-953ed40fc-dev+2026082112
XTDK           3.9.0.1
XRE            5.37.0.0
XCCL           3.2.1.6
```

环境由 `python -m venv --copies` 创建，共用基础 Python 的标准库，第三方包独立
安装。未修改公共 Conda 环境、系统包数据库或宿主机驱动。宿主机 `xpu-smi`
报告驱动 5.0.21.43；上述 XRE 是新环境中的用户态运行时版本。

厂商包来自维护者的 [Torch 2.9 安装说明](https://github.com/baidu/vLLM-Kunlun/blob/v0.25.1-dev/docs/source/installation.md)
及其中的 [XPytorch 安装包](https://klx-sdk-release-public.su.bcebos.com/kunlun2jituan/20260806/xpytorch-cp310-torch290-ubuntu2004-x64.run)。
下载和解包分别使用新的空目录，检查安装脚本后，仅把 `xcudart_package.deb` 的数据
解包到独立环境，并用该环境的 Python 安装厂商 wheels 和 `torch_plugin`。
没有执行原脚本的 `dpkg -i` 或 `XFLAGS --reset`。

厂商 wheel 未声明完整 Python 依赖；导入时需补 `colorama` 和
`hyperparameter==0.5.12`。`hyperparameter==0.6.0` 已移除厂商需要的
`param_scope`，不可直接使用最新版。`torch_plugin` 构建时设置
`CONDA_PREFIX` 为独立环境，使运行时搜索路径指向其中的 `xcudart/lib`。
还在环境的 `lib` 下链接了基础 Python 的 `libpython3.so`。
安装时应使用新目录，已有环境不要重复初始化。

已核对进程加载映射：`libbkcl.so`、`libxpurt` 和 `libxpucuda`
均从上述独立环境加载。设备加法检查返回 4.0，识别到八张卡。

## 模型与数据

- 模型：`Qwen/Qwen3-Reranker-8B`，revision
  `77d193c791ed757ca307ee72715aa132723da912`。
- 数据：`sentence-transformers/t2ranking` 的 `triplet-15`，revision
  `2120acdc3c422d27383db6cd1ad5bc898adca353`；11 个 Parquet，340,111 行。
- 默认目录：`${RETRIEVAL_ROOT}/flagscale-p800-assets/models/Qwen3-Reranker-8B`
  和 `${RETRIEVAL_ROOT}/flagscale-p800-assets/datasets/t2ranking/triplet-15`。
- 按 query hash 划分 train/validation/test，modulo 20，约 90%/5%/5%。
  同一个 Parquet 目录分别传给三个 split，由 loader 筛选。

下载时按部署环境配置网络。离线训练入口设置 `HF_HUB_OFFLINE=1` 和
`TRANSFORMERS_OFFLINE=1`，直接读取本地文件。

## 可复现入口

从仓库根目录启动；先将环境和数据目录变量设为自己的部署位置：

```bash
: "${FLAGSCALE_PYTHON_ENV:?Set an isolated XPytorch environment}"
: "${RETRIEVAL_ROOT:?Set the asset and run workspace}"
export FLAGSCALE_PYTHON_ENV RETRIEVAL_ROOT
bash examples/retrieval/run-p800-fsdp2.sh
```

入口默认选择八张卡，直接通过 `torch.distributed.run` 调用原生 trainer，保留真实
退出码。`FLAGSCALE_PYTHON_ENV` 必填；入口优先加载其中的运行时库。
`CARDS`、`RETRIEVAL_ROOT`、`MODEL_PATH`、`DATA_PATH` 和 `OUT` 可覆盖默认值。
每次运行创建新输出目录并保存完整解析的 `config.yaml`，拒绝覆盖已有目录。
额外参数作为 Hydra overrides 传入。

默认是完整链路的短测试：每卡 batch 1、梯度累积 1、每卡 16 条训练样本、每卡每个
评估 split 4 条样本、一个 epoch、max length 640、lr 1e-7 → 1e-8、warmup 2 步。
`max_samples` 是每 rank/worker 的上限，因此八卡共 16 次优化器更新，每步全局
8 个 triplet（16 个 query/document pair）。这不是完整数据集的三轮收敛结果。
各 rank 使用相同样本上限，避免 streaming epoch 长度不等导致 collective 顺序不一致。

## FSDP2 小模型验证与修复

`examples/retrieval/smoke_fsdp2.py` 使用两层 128×128 权重和一行输出权重，验证
八卡通信、三次 backward/AdamW、fp32 参数和两个 fp32 矩、完整 state dict 的
键与形状、权重实际变化及 strict reload。在设置好独立环境的 PATH、
`CONDA_PREFIX`、`LD_LIBRARY_PATH` 和仓库 `PYTHONPATH` 后运行：

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  examples/retrieval/smoke_fsdp2.py
```

PyTorch 2.9 测试退出码为 0，输出 `FSDP2_PROBE_PASS`。rank 0 三步 loss：
`0.0071356297 → 0.0026536584 → 0.0011024475`。

保留的代码修复是 functional all-gather fallback 的输出形状：应返回沿第 0 维
拼接的张量，不能把结果扁平化。扁平化会让 DTensor 按元素裁剪不均匀分片的
padding，把 `[1, 128]` 权重错误还原为 `[1017]`；大型 embedding 也会受影响。
小模型的一行输出权重覆盖此场景。

PyTorch 2.5 的厂商 split-copy 在 pre-backward all-gather 时还存在参数版本计数
问题。探索阶段验证了临时兼容方式，但升级到 2.9 后无需该补丁，最终代码已移除。

## 真实 8B 短测试结果

输出由 `OUT` 指定；未设置时使用
`${RETRIEVAL_ROOT}/flagscale-p800-runs/fsdp2-fp32-smoke-<timestamp>`。
以下为脱敏后的实验结果，未附带原始日志、配置或环境清单。

| 项目 | 结果 |
|---|---|
| 训练进程退出码 | 0 |
| 优化器更新 | 16 步，8 ranks |
| 全局平均 train loss | 0.4374704957 |
| validation loss | 0.6227476597 |
| test loss | 0.4051435590 |
| epoch checkpoint | `checkpoint-epoch-1/model.safetensors` |
| 最终 checkpoint | `model.safetensors` |
| 每份 fp32 权重文件 | 32,754,238,728 bytes |
| 完成标记 | `training_complete.json`，world_size=8 |

`metrics.json` 记录实际 PyTorch 版本 `2.9.0+cu128`、`parallel=fsdp2`、
`master_dtype=float32` 和全部指标。训练 stdout/stderr 与退出码分别保存在
`train.stdout.log`、`train.stderr.log`、`train.return_code.log`。

最终代码再次通过八卡小模型测试。随后单独重新加载真实 8B checkpoint，检查结果：

- 全部 399 个权重键及张量形状与原模型一致，包括 `[151669, 4096]` 的 embedding
  和 `lm_head`，没有裁剪或展平损坏。
- 抽查保存的 q_proj 权重为 fp32，相对原模型的最大绝对更新为
  `9.760260581970215e-07`，确认参数实际变化。
- Hugging Face loader 的 `missing_keys`、`unexpected_keys`、`mismatched_keys`
  和 `error_msgs` 全部为空。
- 重新以 bf16 加载到单卡后，所有参数有限；前向 logits 形状为 `[1, 151669]`，
  全部有限，最大绝对值 21.125。
- 验证进程退出码为 0，输出 `CHECKPOINT_RELOAD_PASS` 和
  `FINAL_VERIFICATION_PASS`。

证据保存在 `probe.stdout.log`、`probe.stderr.log`、`reload.stdout.log`、
`reload.stderr.log` 和 `reload_verification.json`。另有 `requirements-frozen.txt`、
`source_manifest.json` 和 `parallel.patch`，记录包版本、测试源码 SHA256 和修复。
公共环境的 PyTorch 仍为 `2.5.1+cu118`。

厂商导入钩子会输出缺少 Qwen3.5/Gemma4 可选模型替换目标的 warning，另有
`torch_dtype` 弃用和 torchrun 的 runpy warning；这些日志保留在 stderr，
本次 Qwen3 模型训练退出码为 0。

本次证明短训练、验证、测试和保存链路可运行，尚未验证全数据集三轮训练、
生产 batch size、收敛收益或优化器状态的断点续训。当前 trainer 保存模型权重，
不保存 optimizer/scheduler/RNG resume state。

本地 Python AST、shell 语法、`git diff --check`、新测试脚本的 Ruff lint/format
检查通过。`parallel.py` 全文件 Ruff 仍报告三项原有问题（UP035、RUF100、UP028），
已用 HEAD 内容复查，结果相同，本次未扩展修改这些代码。
