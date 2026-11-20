# 本周总结（2026-11-13 ~ 11-20）：昆仑芯 reranker FSDP2 首次跑通

**一句话**：昆仑芯 P800 上的 Qwen3-Reranker-8B **FSDP2 双卡训练第一次真正跑起来**，
1,170 步 loss 从 0.482 降到 0.270，第一个 checkpoint 已验证为训练后权重可用。

> 之前所有"跑到 2,260 步就崩"的记录，**根因不是 XCCL/硬件**，而是路径、缓存目录和
> batch 配置三件事。下面第二节逐条列出。

---

## 一、当前配置（实际生效，取自 run 的 hydra config）

| 项 | 值 | 说明 |
|---|---|---|
| 模型 | Qwen3-Reranker-8B | 全参微调，原生 yes/no logit 打分 |
| 并行 | **FSDP2** | 2 卡数据并行（dp=2） |
| 权重精度 | **fp32 master** | compute 走 bf16 autocast |
| batch | **16 / 卡** | 2 卡 = **32 行/步**（A100 是 64，只有一半） |
| grad accumulation | 1 | accum=2 实测额外要 2.31 GiB 并 OOM |
| lr / min_lr | **1e-5 / 1e-6** | cosine，warmup 200 步；**与其他平台同值** |
| epochs | 3 | 与共享配置一致 |
| max_length | 640 | p99 471 / max 541，与配置一致 |
| save_steps | **1000** | 新增：中途快照，死一次最多丢约 2.8 小时 |
| 数据 | T2Ranking `triplet-15` | `split_modulo=20` → train 90% |
| 步数 | 9,565 步/epoch | 306,099 行 ÷ 32 |
| 实测步速 | **~9.9 s/步** | epoch 1 约 **26 小时** |
| 显存 | 95 GB / 96 GB | 贴着上限，无余量 |

**关键**：lr 用了和其他平台**相同的 1e-5**，不像 Megatron 路线必须降到 1e-7。
原因是 FSDP2 的 fp32 master 权重解决了 bf16 的分辨率问题。

启动方式（路径全部来自共享盘）：

```bash
source /data_427/jiangxuhui/flagos-retrieval-vendor-env.sh
cd /data_427/jiangxuhui/code/FlagScale
python -m flagscale.run --config-path examples/retrieval/conf --config-name train \
  train=qwen3_reranker_p800 train.parallel=fsdp2 train.master_dtype=fp32 \
  experiment.runner.nproc_per_node=2 train.hardware.device_type=cuda \
  +train.save_steps=1000
```

---

## 二、本周改动

### 2.1 xinyu gao（omer Gao，PR #1，commit `e0169f11`）

`flagscale/train/utils/parallel.py` 的 9 行修复，**这是 FSDP2 checkpoint 能用的前提**：

```python
# 修复前：返回扁平 buffer，DTensor 按「元素」而不是「行」裁剪 padding
out = torch.empty(tensor.numel() * group_size, ...)
work = pg._allgather_base(out.view(group_size, *tensor.shape), tensor, ...)

# 修复后：返回拼接后的形状
out = torch.empty((group_size * tensor.shape[0], *tensor.shape[1:]), ...)
work = pg._allgather_base(out, tensor, ...)
```

**问题**：厂商 `ProcessGroupXCCL` 缺 `allgather_into_tensor_coalesced`，走 shim 的
per-tensor 回退。shim 返回扁平 buffer 时，DTensor 在裁剪各 rank 不均匀分片的 padding
时**按元素裁**，导致 gather 出来的 checkpoint 形状全错（399/399 MISSING）。

同一 PR 还加了：8 卡 fp32 AdamW 探针、checkpoint 探针、可配置的 P800 reranker
启动器，以及把平台文档里的部署路径换成环境变量占位。

### 2.2 我这边（按问题分）

**① 数据分段的三个缺陷**（`fae6cbd7` / `f3505796`）

分段训练原本用 `data.skip_samples` 走不同数据块，但有三个 bug：

| 缺陷 | 后果 | 修法 |
|---|---|---|
| skip 在 shuffle **之前**应用 | 每段重读上一段约 2 万行（实测 166/2000） | 移到 `__iter__` 里、shuffle 之后 |
| `skip_samples` 误传到 eval split | seg2 起 skip 28,800 > 验证集 17k，验证集为空，训练起不来 | 只对 train split 生效 |
| `lr_at` 与实际 scheduler 不一致 | seg1 要 lr=5e-8 < min_lr，第一步就崩 | 逐字对齐 `build_lr_scheduler` |

**② checkpoint 写出后会读成 base 模型**（`062a44a7`，最严重）

`save_model` 复制源目录的附属文件时，排除列表只写了 `model.safetensors`，
于是 base 的 **5 个分片 + 旧索引**也被复制进来。`from_pretrained` 优先读索引，
加载的是 base 权重，**完全忽略刚写出的训练权重**。

实测证据：30 步后 `model.safetensors` 有 621,186,321/621,236,224 个元素与 base 不同，
而索引指向的分片**逐位相同**。

修法：新增 `_is_weight_file()` 排除所有 `.safetensors`/`.bin`/`.pt`/索引，
并显式写出只指向 `model.safetensors` 的新索引。

**③ 退出码被掩盖**（`fae6cbd7` 一部分）

生成的启动脚本是 `bash -c "$cmd; sync"`，返回的是 **`sync` 的状态码**，
所以训练崩溃也报 `exit=0`。修法：捕获 `$cmd` 的退出码再 `sync; exit $rc`。

**④ 周期存盘**（新增 `train.save_steps`）

原来只有 epoch 结束才存盘，一次中途死亡丢掉整个 epoch（实测 3.4 小时白跑）。

**⑤ 路径统一**（`21358d60`）

5 个脚本硬编码 `/data2/jiangxuhui`。迁移到共享盘后第二台机器直接起不来
（它的 `/data2/jiangxuhui` 不存在）。新增 `examples/retrieval/paths.sh`，
所有脚本从 `FLAGOS_DATA_ROOT` 派生路径。

**⑥ P800 batch 例外**（`e4bdd363` / `ccc77974`）

新建 `qwen3_reranker_p800.yaml`，继承主配置只覆盖 `batch_size: 16`，
主配置保持 32 给 A100/PPU。这样启动命令在各平台一致，差异在配置里可见。

---

## 三、遇到的主要困难：昆仑芯很不稳定

### 3.1 真正的根因（都不是硬件）

这周最大的教训：**把四类环境问题误判成了"硬件坏了"，浪费了大量时间**。

| 现象 | 我当时的判断 | 实际原因 |
|---|---|---|
| 训练 1,220 步静默消失，无 traceback、`exit=0` | XCCL 崩溃 | **SSH 会话被 systemd 清理**（`setsid` 不脱离 session cgroup） |
| 探卡 `Failed to initialize runtime libraries` | 驱动坏了/卡坏了 | 我的批量脚本设了 `TMPDIR=/tmp`，**覆盖了 vendor env** |
| `torch.mm` 挂死 | 卡执行不了 kernel | **XBLAS JIT 缓存目录不存在**（`Failed to create lock file`） |
| batch 32 OOM "需 11.57 GiB，20.9 GiB 空闲" | 显存不够 | **碎片**，不是容量；vendor 分配器无 `expandable_segments` |

### 3.2 真实的不稳定项

**① 会话依赖（已解决）**
`ssh 'setsid nohup ...'` 起的进程仍在 SSH session 的 cgroup 里，断开即被杀。
**必须用 tmux**。这一条已写进 AGENTS.md §九。

**② 共享盘读取慢**
从共享盘加载 8B 权重，**每个分片 45 秒**（本地盘是 3.5 秒），5 个分片约 4 分钟。
每次重启都要付这个成本。

**③ 多租户抢卡**
两台机器上都有别人的 benchmark（`x-benchmark` docker 容器，跑了 16 小时产出 0 个结果），
容器内能看到全部 8 张卡的设备节点，**没有卡隔离**。判断「这台机器能不能用」
只能靠实测，`xpu-smi` 显示 0 MiB / 0% 不代表可用。

**④ 显存贴着上限**
95 GB / 96 GB。batch 16 是目前唯一能跑的配置，**没有余量应对更长的样本**。

**⑤ 共享盘代码会让两台不一致**
两台挂同一个 Lustre 卷。`sync.sh push` 后面跟 `git clean -fd` 会删未跟踪文件，
所以**只能对一台推**，推两次会互相删文件。

### 3.3 诊断顺序（血的教训）

遇到"训练静默死亡"，按这个顺序查，**顺序反了会把运维问题当硬件问题**：

```bash
# 1. 先查会话是否被杀
journalctl --since "..." | grep 'scope.*Deactivat'
# 2. 再查内核层
sudo dmesg -T | grep -E 'KL_XID|kl3_wait_for_noc_idle'
# 3. 最后才怀疑卡：分层探卡
#    alloc(纯分配) -> fill(常量) -> randn(RNG) -> mm(矩阵乘)
#    这四步分开测才能定位是哪个 kernel 挂了
```

---

## 四、验证结果

### 4.1 训练

| 步数 | loss | lr |
|---|---|---|
| 5 | 0.4818 | 3.0e-7（warmup） |
| 305 | 0.3430 | 1.0e-5（峰值） |
| 1,060 | 0.2774 | 1.0e-5 |
| **1,170** | **0.2703** | 1.0e-5 |

无发散，无 NaN。

### 4.2 checkpoint（step 1000）

| 检查 | 结果 |
|---|---|
| 文件 | 14 个，**无 base 的 5 个分片** ✅ |
| 索引 | 只指向 `model.safetensors` ✅ |
| 张量 | 399 个完整 ✅ |
| **权重是否被训练** | `embed_tokens` **621,189,768 / 621,236,224** 个元素变化，`max\|d\|=1.9e-3` ✅ |

**这个 checkpoint 可以直接拿去 `evaluate_retrieval.py` 评测或作为下一段热启动。**

---

## 五、待办

1. **epoch 1 跑完**（约 26 小时）后测 nDCG@10，与 A100 / PPU 对比
2. 1,000 步的中间 ckpt 可先评一次，看趋势
3. 显存 95/96 GB 贴着上限，若出现 OOM 需要降 batch 或加卡
4. `sync.sh` 对共享盘只能推一台，这个约束要写进操作手册

---

## 附：共享盘布局（本周迁移）

```
/data_427/jiangxuhui/           # Lustre，两台 P800 共用，20 T
├── models/          26 G   Qwen3-Reranker-8B / bge-m3 / clip
├── datasets/        26 G   t2ranking / MSCOCO / Flickr30K
├── ckpts/          246 G   klx-pp4-3ep + 两个 save 验证样本 + 本次 run
├── megatron-reranker 124 G
├── code/                    FlagScale / Torch-FL（推一台两台生效）
└── envs/            12 G    ads + retrieval-vendor（含 XPytorch）
```

验证共享的方法：两台 `stat -c %i` 同一文件，**inode 相同**才算。
已实测 `train_retrieval.py` 两台都是 `144115205423503593`。
