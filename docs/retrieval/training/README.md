# training/ — 训练与并行方案

跨芯片跑训练的**操作手册**。并行方案的选型依据在仓库根的
[`MULTI-GPU-SCHEME.md`](../../../../MULTI-GPU-SCHEME.md)。

| 文档 | 用途 |
|---|---|
| [**19-adaptation-report**](19-adaptation-report.md) | **给甲方的适配说明**：标准流程、四平台各自改了什么、工作量汇总、遗留问题 |
| [**18-environment-and-scripts**](18-environment-and-scripts.md) | **换机器上手手册**：四平台环境实况、Python 入口、路径约定、训练/评测/分析脚本用法 |
| [15-launch-commands](15-launch-commands.md) | **各芯片的启动命令原文**。每台机器只有一条验证过的路径，附每个踩过的坑与报错 |
| [12-flaggems-integration](12-flaggems-integration.md) | FlagGems 开关怎么用、怎么验证、为什么实测还是不快 |

## 并行方案速查

| 平台 | runtime | 并行 | 备注 |
|---|---|---|---|
| A100 | 原生 CUDA | FSDP2 2 卡 | 基准 |
| PPU ZW810E | Torch-FL `flagos` | FSDP2 2 卡 | 必须进容器，卡号在容器内重新编号 |
| 昆仑芯 P800 | 厂商 XPytorch（`device_type=cuda`） | **Megatron PP=4** | PP=2 装不下整步，被迫用小 micro-batch 导致训练崩 |
| 昇腾 910C | torch_npu | FSDP2 4 卡 | batch 16/卡（显存 64 GB） |

> ⚠️ **AGENTS.md §八**：命令行只传模型名、并行方式、设备、输出位置。
> 超参（batch/lr/epoch/max_length/数据路径）一律来自 `conf/train/*.yaml`。
