# 21 · A100 基准代码讲解

**逐个模块讲清 A100 基准路径的实现**：模型、数据、训练循环、存盘与续训、评测。
这是所有平台对齐的目标版本——其它平台的差异见 [19](19-adaptation-report.md)。

代码入口：`flagscale/train/train_retrieval.py`（1,365 行）+ `flagscale/models/retrieval/` +
`flagscale/train/datasets/retrieval.py`。

---

## 一、整体架构

```
配置 (conf/train.yaml + conf/train/<model>.yaml)
  ↓
train_retrieval.train()
  ├── init_distributed()          设备/进程组/平台
  ├── build_retrieval_model()     模型适配器（HF 权重）
  ├── build_split_loaders()       数据集 + DataLoader
  ├── apply_fsdp2 / DDP           并行化
  ├── build_optimizer()           AdamW
  ├── build_lr_scheduler()        cosine + warmup
  └── for epoch:
        ├── train_epoch()         前向/反向/优化
        ├── evaluate_epoch()      val loss
        └── save_model()          每 epoch 一次
  ↓
evaluate_retrieval.py            独立评测脚本（单进程，不依赖 torchrun）
```

**两条路径**：`train()` 里按 `train.parallel` 分岔——
`ddp`/`fsdp2` 走原生流程（本文），`megatron` 走 `_train_megatron()`（昆仑芯，见另一文档）。

---

## 二、模型层

### 2.1 适配器模式：一个模型一个 wrapper

```
flagscale/models/retrieval/
├── base.py                  RetrievalModel 抽象基类
├── registry.py              @register_retrieval_model 装饰器 + build_retrieval_model()
├── bge_m3.py
├── clip_vit_large.py
├── qwen3_embedding.py
├── qwen3_reranker.py
└── qwen3_vl_embedding.py
```

`RetrievalModel` 定义六个必须实现的接口：

| 方法 | 作用 |
|---|---|
| `prepare_embedding_inputs(inputs)` | 把文本/图文列表变成模型输入 |
| `prepare_reranker_inputs(pairs)` | 把 (query, doc) 对变成输入 |
| `forward(features)` | 前向，返回 `{"sentence_embedding": ...}` 或 `{"scores": ...}` |
| `save_pretrained(dir)` | 存盘（HF 布局）|
| `freeze_backbone()` / `enable_gradient_checkpointing()` | 训练开关 |
| `disable_cache()` | 关 KV cache（推理时不需要）|

**为什么用适配器**：训练循环、数据加载、loss、并行化全部与模型无关，
换模型只加一个 wrapper，不改主流程。新增模型 = 写一个类 + 注册。

### 2.2 三种 loss 对应三种任务

`flagscale/train/losses/retrieval.py`：

| task | loss | 说明 |
|---|---|---|
| `embedding` | 对比学习（in-batch negatives）| `text_embeddings @ text_embeddings.T / temperature` |
| `reranker` | **BCE with logits** | 每行产出 1 正 + N 负 |
| `clip` | 对称对比 | 双向 cross entropy 取平均 |

### 2.3 Reranker 的打分方式（关键）

`qwen3_reranker.py` 用**模型自己的 yes/no 打分**，不用随机初始化的 head：

```python
final_logits = logits[rows, positions]           # 最后一个 token 的 logits
scores = final_logits[:, self.true_token_id] \
       - final_logits[:, self.false_token_id]    # yes - no
```

配置里 `use_score_head: false`。原因：随机初始化的标量头会**丢掉预训练的
reranking 能力**——带 head 时 loss 高于 ln2 随机基线，用原生 logit 打分
起点就是 BCE 0.43（acc 0.81）。

**`false_token_id` 必须是 2152（`'no'`），不是 9692。** 9692 解码是 `'icken'`，
两个无关 logit 相减仍是有限数，所以不报错，只是模型永远学不到东西。
这个 bug 在昆仑芯路径上造成过很大代价，见 [19](19-adaptation-report.md) §2.2(d)。

### 2.4 一行产出几个 pair

`reranker_loss` 里，**每行**产出：

```
pairs.append([anchor, positive])      labels.append(1.0)
for key in negative_keys:
    pairs.append([anchor, batch[key]]) labels.append(0.0)
```

`num_negatives: 1` 时每行 **2 对**。所以 `batch_size: 32` → FSDP2 2 卡 →
64 行 = **128 对/步**。这个换算在所有吞吐口径里都要用。

---

## 三、数据层

### 3.1 流式 + 分片，不落盘

```
ParquetTripletDataset  /  ClipTarDataset   都是 IterableDataset
  ↓ 按 (dp_rank, worker_id) 分片
DataLoader(batch_size, num_workers=0, collate_fn=retrieval_collate)
```

**为什么用 `IterableDataset` 而不是 `Dataset`**：数据集 1.1 GB / 34 万行，
`Dataset` 需要随机访问，流式读取更适合；且天然支持"边读边分片"。

### 3.2 三重处理管线

```python
__iter__():
    stream = _buffered_shuffle(_iter_rows(), shuffle_buffer, seed + epoch*7919 + rank)
    if length_sort_window > 0:
        return _length_sorted(stream, window)
    return stream
```

| 阶段 | 作用 | 配置 |
|---|---|---|
| **split 划分** | 按 query 哈希分 train/val/test（90/5/5）| `split_modulo: 20` |
| **分片** | `accepted % world_size == rank` | 自动（dp_rank × workers）|
| **打乱** | 20000 条的缓冲池 | `shuffle_buffer: 20000` |
| **长度排序** | 窗口内按长度排，减少 padding | `length_sort_window: 256` |

### 3.3 两个必须理解的设计

**（a）split 按 query 分组，不按行**

```python
split_accepts(row_index, split, modulo, group_key=str(row.get("anchor")))
```

T²-Ranking 一个 query 有约 14 行。按行索引分桶会把同一 query 的行打散到
train/test 两边——实测 **89.4% 的 test query 在训练集里出现过**。按 query
哈希分桶后每行都跟着自己的 query 走。

**（b）按 data-parallel rank 分片，不按全局 rank**

```python
if _ps.model_parallel_is_initialized():
    rank = _ps.get_data_parallel_rank()
    world_size = _ps.get_data_parallel_world_size()
```

PP 下每个 stage 必须消费**同一批样本**（stage1 打分的是 stage0 算出的 hidden）。
按全局 rank 分片会让两个 stage 拿到不同数据——P800 上实测同一 prompt
经 p=2 打分 4.55、经 HF 打分 1.25，而单 stage 同权重是 1.31（正确）。

### 3.4 `num_workers` 默认 0

配置里没设，DataLoader 单进程解码。**这是已知的性能短板**：tokenize + 解码
在主进程串行，GPU 会等。但在 A100 上 tokenize 只占 6.7%（见
[20](../analysis/20-slow-op-profiling.md)），所以没优先处理。

---

## 四、训练循环

### 4.1 `train_epoch()` 的核心：梯度累积 + 选择性 sync

```python
stream = iter(loader)
batch = next(stream, None)
while batch is not None:
    nxt = next(stream, None)                       # 一步前瞻
    with autocast_context(device, dtype_name):
        raw_loss = loss_for_batch(...)
        loss = raw_loss / grad_accum               # 归一化
    is_window_end = (micro_count + 1) >= grad_accum or nxt is None
    with nullcontext() if is_window_end else gradient_sync_disabled(model):
        loss.backward()
    ...
    if micro_count < grad_accum:
        continue                                   # 累积窗口未满
    clip_grad_norm_(trainable, max_grad_norm)
    optimizer.step(); zero_grad(); lr_scheduler.step()
```

三个要点：

**（1）一步前瞻**：`nxt = next(stream, None)` 提前取下批。
最后一个 micro-batch 即使不满累积窗口也必须 sync（`or nxt is None`），
否则那一步的梯度没跨 rank reduce。

**（2）只在窗口末尾 sync 梯度**：`gradient_sync_disabled(model)` 包住中间步。
每步都 sync 会把集合通信量乘 `grad_accum`，跨机器时是墙钟的主要来源。

**（3）loss 先除 `grad_accum`**：保证累积后的梯度是平均而非求和。

### 4.2 LR schedule 有一个"未知总步数"的状态

数据是流式的，**跑完一个 epoch 才知道有多少 optimizer step**。所以：

```python
state = {"total_steps": 0, "known": False}      # 初始未知
...
def factor(step):
    if not state["known"]:
        return 1.0                               # 未知时保持在 base_lr
    progress = (step - warmup) / (total - warmup)
    return floor + (1 - floor) * (0.5 * (1 + cos(pi * progress)))
```

epoch 1 结束时，调用方量出真实步数并 re-anchor：

```python
refined = max(1, round(steps_done / epoch * epochs))
lr_sched_state["total_steps"] = refined
lr_sched_state["known"] = True
```

**为什么不能给一个占位值**：早期版本传了个 placeholder，导致 `known=True`、
re-anchor 被跳过、lr 在第 2 步就衰减到 min_lr——**每个 run 全程都在 min_lr 上训练**。

**schedule 不是可选项**：不加 schedule 的 3 epoch 8B reranker（恒定 1e-5）
train 0.042 / val 0.377（9 倍差），R@5 比 base 低 1.5 点。

### 4.3 自动混合精度

```python
compute_dtype = {"fp16": float16, "fp32": float32}.get(dtype_name, bfloat16)
```

- 模型参数：`master_dtype`（配置 `fp32`）
- 前向/反向：autocast 到 `bf16`
- 优化器状态：fp32

> **这是 A100 基准的关键设定**：参数本体是 fp32。昆仑芯的 Megatron 路径参数是
> bf16，导致 lr=1e-7 的更新低于 bf16 分辨率（见 [19](19-adaptation-report.md) §4.3）。

---

## 五、并行化

### 5.1 FSDP2 的三个处理

```python
# 1. 先在 CPU 建模型
model_device = torch.device("cpu") if parallel_mode == "fsdp2" else device
model, processor = build_retrieval_model(..., model_device)

# 2. 分片
model = apply_fsdp2(model, device_type, world_size,
                    compute_dtype=bf16, reduce_dtype=bf16)

# 3. 只把本地 shard 移到设备
model.to(device=device, dtype=master_dtype)
```

**为什么先在 CPU 建**：每个 rank 先加载完整 fp32 8B 约 33 GB，之后还要放
sharded state 和 all-gather 缓冲，64 GB 卡直接 OOM。CPU 建完再分片，
只有本地 shard 上设备。

### 5.2 两个 shim（适配厂商缺失的 API）

`flagscale/train/utils/parallel.py`：

| shim | 解决什么 |
|---|---|
| `install_coalesced_allgather_shim()` | 厂商 `ProcessGroupNCCL` 缺 `allgather_into_tensor_coalesced`，FSDP2 用它做参数组 all-gather |
| `full_state_dict()` | FSDP2 的 DTensor 收集成普通张量（存盘用）|

启动日志会打印 `coalesced all-gather shim installed on ProcessGroupNCCL`。

### 5.3 `parallel.py` 的其他职责

- `gradient_sync_disabled(model)`：累积窗口内不 sync
- `fsdp2_available()`：检查 torch 版本是否支持 composable FSDP
- oversized leaf 拆分：`lm_head` / `embed_tokens` 单独分片（避免 all-gather 出 2.3 GiB 大块）

---

## 六、存盘

### 6.1 `save_model()` 的三条路径

| 情形 | 写法 |
|---|---|
| DDP（单卡）| `save_pretrained()` 直接写 HF 布局 |
| **FSDP2** | 先 `full_state_dict()` 收集（集合通信，**所有 rank 都要调**），再由 rank 0 写单文件 |
| Megatron | `stage{K}.pt` 每 rank 一个文件 |

### 6.2 FSDP2 存盘踩过的三个坑（都修了）

**这三个缺陷让每一份 FSDP2 checkpoint 都加载不出真实权重，且存盘时不报错。**

| 缺陷 | 现象 | 修法 |
|---|---|---|
| 张量带 `backbone.` 前缀 | HF 报 399/399 MISSING | `removeprefix("backbone.")` |
| 张量被展平成 1-D | `[12288,4096]` 存成 `[50331648]` | 按参数量 reshape |
| 最大两个张量多 4095 个尾部元素 | `621,240,319` vs 应为 `621,236,224` | trim |

**后果的量化**：`from_pretrained` 载入随机初始化模型，评测时 base BCE 0.429、
"trained" 0.942（比 ln2=0.693 还差，正是随机权重的样子）。**所以 val 0.72→1.04
不是过拟合，是评了个随机网络。**

### 6.3 `training_complete.json`

只有 `final=True` 时才写，用来区分"完成的模型"和"中间 epoch 快照"。

### 6.4 存盘时机的教训

**存盘要写在任何可能失败的步骤之前。** 早期版本是 `train → evaluate → save`，
评测一崩（昆仑芯上崩过）整个 epoch 的权重视同丢失——实测损失 16 小时。
现在是 `train → save → evaluate`。

---

## 七、续训

### 7.1 现状：只能恢复权重

`save_model()` **只存参数**。checkpoint 目录里没有：

- ❌ 优化器状态（AdamW 的一阶/二阶矩）
- ❌ LR schedule 位置
- ❌ RNG 状态

**所以"续训"是权重热启动，动量从零开始。** 这是一个已知限制，不是 bug——
要做到完整续训需要在 `save_model` 里加 optimizer state_dict。

### 7.2 现有的补偿做法

`examples/retrieval/train-reranker-resume-a100.sh`：把 lr 设成中断时 cosine
**应该在的位置**，让"动量清零"的扰动最小。

```bash
# 原 3 epoch 计划：total 14,382 步
# epoch 1 结束在 step 4,794
# 该位置 cosine = 7.86e-6（日志实测 7.860e-06，吻合）
train.optimizer.lr=7.86e-6 train.epochs=2
```

### 7.3 一个口径问题

续训脚本命令行传 `epochs=2`，这违反 AGENTS.md §八（epochs 是禁改项）。
理由：这是**续训**不是换配方——epoch 1 已完成并作为起点，剩 2 个 epoch。
这类例外需要在报告里标注。

---

## 八、评测

### 8.1 独立脚本，不依赖训练进程

`examples/retrieval/evaluate_retrieval.py`：

```bash
python examples/retrieval/evaluate_retrieval.py \
  --model-type bge_m3 --task embedding \
  --model-path <原始权重或 checkpoint> \
  --data-path <数据集> --split test \
  --num-negatives 15 --batch-size 32 --device cuda \
  --output metrics.json
```

**单进程设计**（不需要 torchrun）：一次评一个模型，跑完退出。
8B reranker 单卡放不下时这条路走不通（昆仑芯/PPU 的缺口）。

### 8.2 两种评测协议

| task | 协议 |
|---|---|
| `embedding` | 每 query 的 positive 对 N 个 hard negative 排序 |
| `reranker` | 每 query 的 (positive + N negatives) 逐对打分再排序 |
| `clip` | **in-batch 全库检索**：5000 文本 × 5000 图像对称排序 |

**CLIP 必须是全库**：早期版本每 batch 内部排序，候选池是 batch size（64）
而不是测试集（5000），召回率虚高两个数量级。

### 8.3 指标与论文口径

`tools/recompute_metrics.py` 可从 `*.predictions.json` 重算，不用重跑模型：

```bash
python tools/recompute_metrics.py preds.json --cutoffs 1 5 10 20 --ndcg
```

| 模型 | 论文主指标 | 我们报 |
|---|---|---|
| BGE-M3 | nDCG@10 + Recall@100 | nDCG@10（Recall@100 受候选池限制报不出）|
| CLIP | R@1/5/10（**无 nDCG**）| R@1/5/10 |
| Reranker | nDCG@10（FollowIR 用 p-MRR）| **协议不对，报不出** |

> 详见 [papers/README](../papers/README.md)——**三个模型主指标不同，不取并集**。

### 8.4 推理吞吐的测量

`InferenceClock`（在 `evaluate_retrieval.py` 里）：只计模型前向，
前后各 `synchronize` 一次，数据加载和 CPU 排序都在计时区间外。

> ⚠️ **必须在空闲卡上单独测。** 并行跑 BGE+CLIP 时 CLIP 只测到 100.3 样本/s，
> 单独测是 461.1——**低估 4.6 倍**，因为两进程抢 CPU 导致 GPU 喂数据卡住。

---

## 九、可复现性

### 9.1 种子的三层设置

```python
seed_everything(seed, rank, device, platform, deterministic=False)
```

- 全局 seed（默认 42）
- 每 rank 偏移：`seed + rank`
- DataLoader worker：`seed + worker_id * 7919`

启动日志打印 `reproducibility: seed=42 per_rank=42 deterministic=False world_size=2`。

**`deterministic=False` 是刻意的**：完全确定性会让 cuBLAS 用慢算法，
8B 模型上代价太大。所以"同种子可复现"是指**数据顺序和初始化**，不是逐位一致。

### 9.2 配置记录

每次 run 的完整配置留在 `<exp_dir>/hydra/.hydra/config.yaml`。

> **自检**：跑完看这个文件，确认 `data.path` / `batch_size` / `epochs` /
> `max_length` 与仓库配置逐字一致。不一致就停下来查——这是数据泄漏最常见的来源。

---

## 十、一句话总结

**A100 基准路径的设计原则：尽可能"素"。**

- 不装厂商插件（避免差异归因不清）
- 参数本体 fp32（保证 lr 生效）
- 并行用标准 FSDP2（不引入 Megatron 的复杂度）
- 训练循环与模型无关（适配器模式）

其它平台的所有适配，都是在这套基准上做"绕开厂商缺陷"的减法。
