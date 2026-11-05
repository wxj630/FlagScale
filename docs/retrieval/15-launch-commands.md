# 15 · 各芯片启动命令与踩过的坑

**这份文档的目的：每台机器的启动路径只有一条是验证过的，照抄这里，不要现编。**
现编命令踩过的坑都在下面，每条都附实测报错。

## 零、通用规矩（三条，违反任一条都会浪费一次启动）

1. **命令行只传选择项与环境项**：`train=<model>`、`parallel=`、`device_type=`、
   `nproc_per_node`、`output_dir`、`exp_dir`。
   batch/lr/epoch/max_length/数据路径**一个都不要写**——它们来自
   `conf/train/<model>.yaml`，抄到命令行就会与配置分叉（AGENTS.md §八）。
   唯一例外是 AGENTS.md 登记的平台数值例外（昆仑芯 reranker 的 lr=1e-7）。
2. **输出目录每次换新的**，否则新 run 会覆盖旧 checkpoint。
   2026-11-05 就发生过：A100 的 reranker run 写回
   `a100-fp32master-3epoch/qwen3_reranker`，与 10-27 那次重名。
3. **改完代码先 `./sync.sh push <platform>`，再启动。**
   远端没同步就等于跑旧代码（这条也踩过两次，见
   `Notes/` 里 sync-before-measure 的记录）。

## 一、A100（`bitahub_2gpus`，2×A100-80GB）

**运行方式**：不用容器，直接用 `sentence-transformers` 的 venv。

```bash
# 在远端 /flagos-search-codes/FlagScale 下
export FLAGSCALE_HOME=/flagos-search-codes/FlagScale
export FLAGSCALE_PYTHON_ENV=/flagos-search-codes/sentence-transformers/.venv
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

bash examples/retrieval/run-a100.sh qwen3_reranker 0 0,1
```

要点：

| 项 | 值 | 说明 |
|---|---|---|
| Python | `/flagos-search-codes/sentence-transformers/.venv/bin/python` | 必须显式设 `FLAGSCALE_PYTHON_ENV`，否则 `before_start` 的 `source` 找不到 |
| `RERANKER_CARDS` | `2` | reranker 必须 FSDP2 双卡；单卡放不下 8B 的 fp32 优化器状态 |
| 输出 | `run-a100.sh` 内置 `$OUT` | **重跑前改脚本里的 `OUT`**，否则覆盖 |
| 已踩坑 | `expandable_segments` 必开 | 不开会在 FSDP2 gather 时 OOM |

## 二、昆仑芯 P800（`kunlunxin`）

**运行方式**：宿主机直接用 `envs/retrieval-vendor` 的 Python（厂商 Torch）。

```bash
source /data2/jiangxuhui/envs/retrieval-vendor/bin/activate
cd /data2/jiangxuhui/FlagScale
export PYTHONPATH=/data2/jiangxuhui/FlagScale:/data2/jiangxuhui/megatron-reranker/Megatron-LM-FL
export CUDA_VISIBLE_DEVICES=0,1 XPU_VISIBLE_DEVICES=0,1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export XMLIR_BMM_DISPATCH_VALUE=0        # 必须：否则 seq>=288 的 bmm JIT 崩

python -m flagscale.run \
  --config-path examples/retrieval/conf --config-name train \
  action=test train=qwen3_reranker \
  train.parallel=megatron train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=2 \
  train.hardware.device_type=cuda train.hardware.distributed_backend=auto \
  train.pipeline_model_parallel_size=2 \
  train.stage_dir=/data2/jiangxuhui/megatron-reranker/work/pp2-8b-new \
  train.optimizer.lr=1e-7 train.optimizer.min_lr=1e-8 \
  train.output_dir=<新目录> experiment.exp_dir=<新目录>/logs
```

要点：

| 项 | 值 | 说明 |
|---|---|---|
| `device_type` | **`cuda`**（不是 flagos） | 昆仑芯走厂商 torch，`flagos` 是 PPU 的路径 |
| `XMLIR_BMM_DISPATCH_VALUE` | `0` | 不设 → `bmm_one_loop` JIT 在 seq≥288 崩 |
| `Megatron-LM-FL` 必须在前 | PYTHONPATH 第一段 | 否则 import 到错版本 |
| lr | **1e-7** | AGENTS.md §八 登记的 P800 例外，1e-5/1e-6 都发散 |
| 卡 | 跑前逐卡探活 | 旧机卡 1 硬件坏；新机 8 卡全好 |
| 已踩坑 | 存盘顺序、evaluate 分片、0 元素张量 | 三个 bug 已修（`c2b17ec4`/`78192fac`/`c2d59c95`） |

## 三、PPU ZW810E（`pingtouge`）

**运行方式**：**必须在容器 `flagos-retrieval` 内**。宿主机 Python 是 3.6.8 且无 torch。

### 3.1 进容器

```bash
# 本机（需先连 VPN）
PW=$(sed -n 's/^PPU_PASSWORD=//p' FlagScale/.env | tail -1)
printf '%s\n' "$PW" | ssh pingtouge 'read -r p; printf "%s\n" "$p" | sudo -S -p "" docker exec -it flagos-retrieval bash'
```

> `sudo` 密码 = `.env` 里的 `PPU_PASSWORD`（与 SSH 密码相同）。
> 注意 `docker exec -i` 会占用 stdin，此时 `sudo -S` 读不到密码——
> 用 `docker cp` 传文件，不要用管道灌脚本。

### 3.2 启动（容器内）

```bash
cd /workspace/FlagScale
export CUDA_VISIBLE_DEVICES=<单卡号>       # bge/clip 单卡
export FLAGSCALE_PYTHON_ENV=/usr/local     # 容器里没有 venv
export PYTHONPATH=/workspace/FlagScale:/workspace/Torch-FL

bash examples/retrieval/run-ppu-three.sh clip_vit_large
```

要点：

| 项 | 值 | 说明 |
|---|---|---|
| `GEMS_VENDOR` | `thead` | 不设 → `FlagGems vendor detection failed for FLAGOS_ACCELERATOR='ppu'` |
| `NCCL_SOCKET_IFNAME` | `bond0` | 不设 → `NCCL ERROR Bootstrap : no socket interface found` |
| `FLAGSCALE_PYTHON_ENV` | **`/usr/local`** | 默认值指向 A100 的路径，容器里不存在 |
| `PYTHONPATH` | 必须含 **`/workspace/Torch-FL`** | `device_type=flagos` 需 import `torch_fl`，见下 |
| 卡 | `ppu-smi` 查空闲 | 16 张卡常被别人占满 |
| reranker | `nproc_per_node=2`（FSDP2） | 单卡放不下 8B |

### 3.3 ⚠️ 尚未解决的坑：`device_type=flagos` 找不到 torch_fl

**现象**（2026-11-05，`run-ppu-three.sh clip_vit_large`）：

```
RuntimeError: device_type=flagos was requested, but a Torch-FL platform is
unavailable. Import torch_fl before torch and use its matching PyTorch 2.10.x
environment.
  File "flagscale/train/train_retrieval.py", line 118, in init_distributed
```

**已验证的事实**（手动执行时一切正常）：

```
PYTHONPATH=/workspace/FlagScale:/workspace/Torch-FL python3 /tmp/probe.py
  torch_fl: /workspace/Torch-FL/torch_fl/__init__.py
  device_type check: platform = flagos
```

`torch_fl` 确实装在 `/workspace/Torch-FL/`，手动带上路径就能 `get_platform() == flagos`。

**根因**：`flagscale.run` 生成的
`.../logs/scripts/host_0_localhost_run.sh` 会**重写** PYTHONPATH：

```bash
# 第 3 行：before_start（已加 Torch-FL 探测）
{ source ...activate; } 2>/dev/null || true; for d in "/workspace/Torch-FL" ...; do ... done
# 第 14 行：runner 自己重写
export PYTHONPATH=/workspace/FlagScale:/workspace/FlagScale/flagscale/train:${PYTHONPATH}
```

第 3 行的循环在 `cd /workspace/FlagScale`（第 12 行）**之前**执行，
`$PWD` 还是调用者的目录；`/workspace/Torch-FL` 是绝对路径本该命中，
但实测子进程仍拿不到——**这一点没查清**，所以当前 PPU 的 bge/clip
用 `flagscale.run` 起不来。

**已排除的猜想（都实测过，不要再试）**：

| 猜想 | 实测 | 结论 |
|---|---|---|
| `torch_fl` 没装 | `python3 -c "import torch_fl"` → `/workspace/Torch-FL/torch_fl/__init__.py` | ❌ 装了 |
| PYTHONPATH 少了 Torch-FL | `PYTHONPATH=/workspace/FlagScale:/workspace/Torch-FL python3 probe.py` → `platform = flagos` | ❌ 路径对就能成功 |
| `-m` 模式与文件路径有别 | 两种都试，都失败 | ❌ 不是这个 |
| numpy 先于 torch_fl 被 import | 复现 train_retrieval 的 import 顺序，成功 | ❌ 不是这个 |
| `get_platform` 导入失败 | `T.get_platform is None` → `False` | ❌ 导入正常 |
| `get_platform()` 抛异常 | 直接调用 → `call OK: flagos` | ❌ 调用正常 |
| torchrun 丢了环境变量 | 子进程内打印 → `FLAGSCALE_DEVICE='flagos'` | ❌ 环境在 |
| runner 重写 PYTHONPATH | 生成的脚本第 14 行是 `...:${PYTHONPATH}`，保留了 | ❌ 没丢 |

**已定位到确切失败点（2026-11-05，在真实失败的进程里插桩得到）**：

```
PROBE3 PLATFORMS= [] torch.flagos= True
PROBE EXC: RuntimeError No available platform detected.
           Ensure CUDA, NPU (torch_npu), or MUSA (torch_musa) is installed.
```

即：`torch.flagos` **存在且 is_available() 为 True**，但
`flagscale.platforms.platform_register.PLATFORMS` 是**空字典**，
于是 `get_platform()` 走到最后一行 `raise RuntimeError`。

`register_platforms()` 在 `flagscale/platforms/__init__.py` 导入时执行一次，
之后不重建。它逐个调 `PlatformFlagOS().is_available()` →
`torch.flagos.is_available()`。真实运行里这一步返回了 False（或抛
`AttributeError` 被吞掉），所以 `PLATFORMS` 没有任何条目。

**但同一台机器、同一环境、手动执行时 `PLATFORMS == ['flagos', 'cuda']`。**
下面这些都单独验证过、全部正常，所以问题不在它们身上：

| 验证项 | 结果 |
|---|---|
| `torch_fl` 可导入 | ✅ `/workspace/Torch-FL/torch_fl/__init__.py` |
| `torch.flagos` 属性存在 | ✅ `True` |
| `torch.flagos.is_available()` | ✅ `True` |
| `PlatformFlagOS().is_available()` | ✅ `True` |
| 手动 `register_platforms()` | ✅ `PLATFORMS = ['flagos','cuda']` |
| 导入顺序（torch_fl 先于 platforms） | ✅ 已用 `__import__` 钩子确认 |
| 模块顶部 `sys.modules` 状态 | ✅ 两者均未加载 |
| `numpy` 先于 torch_fl | ✅ 不影响 |
| `sys.path[0]` 与真实路径一致 | ✅ 在 `flagscale/train/` 内跑也正常 |
| torchrun vs 直跑 | ✅ 两种都失败（所以不是 torchrun） |
| `-m` vs 文件路径 | ✅ 两种都失败 |

**结论：失败只出现在 `train_retrieval.py` 这一个文件被真正执行时，
任何等价的最小复现脚本都成功。根因尚未确定。**

**下一步建议**（必须用二分法，不要再猜）：
1. 在 `train_retrieval.py` 的 `init_distributed` 里把
   `PlatformFlagOS().is_available()` 的**异常**打印出来（当前被
   `except (AttributeError, RuntimeError): return False` 吞掉）。
2. 在同一位置打印 `flagscale.platforms.platform_register.PLATFORMS` 的
   `id()`，与模块级 `id()` 对比，确认是不是同一对象（可能是重复导入）。
3. 用 `python -X importtime` 对比真实运行与 probe 的导入时间线。

**可用的替代路径**：`evaluate_retrieval.py --device flagos` 在 PPU 上**成功过**
（2026-11-05 测推理吞吐：BGE 74.1、CLIP 270.9 样本/s）。它是 `python` 直跑、
不用 torchrun。所以 **PPU 的 flagos 推理路径可用，只有 torchrun 训练路径不可用**。

**下一步建议**（给接手的人）：
1. 在 `init_distributed` 第 110 行前插入
   `print("DEBUG", type(T.get_platform), T.get_platform())`，
   在**真实失败的那次运行里**看返回值——静态复现已经证明无效，必须在真实现场看。
2. 检查 torchrun 子进程的 `sys.path` 是否与直跑不同（`runpy` 会改变 `sys.path[0]`）。
3. 若短期必须跑，可考虑让 PPU 的 bge/clip 也走 `python` 直跑
   （`world_size=1` 时不需要 torchrun）。

## 四、昇腾 / 寒武纪

未配置可直连的 host，`sync.sh` 里 `ascend` 只是占位（需 `ASCEND_REMOTE_ROOT`）；
寒武纪没有 host 别名。见 `AGENTS.md` §二。

## 五、这张表要维护

新增平台或改启动方式时，**先更新本文件再动手**。理由：
2026-11-05 为了在 PPU 上重训 CLIP，我在"猜 PYTHONPATH 该放哪"上
连续试了 6 次启动，每次都花 15-25 秒等失败 + 读日志；如果当时有这份文档，
第一次就会看到"容器内手动带 `/workspace/Torch-FL` 可行、但 `flagscale.run`
的脚本会覆盖它"，直接走绕过方案。
