# results/ — 实验结果

各平台的**精度与吞吐**实测，以及支撑这些数字的原始 JSON。

| 文档 | 平台 | 要点 |
|---|---|---|
| [07-a100](07-a100.md) | **A100（基准）** | 所有对比的分母 |
| [08-kunlunxin](08-kunlunxin.md) | 昆仑芯 P800 | 含 reranker 收敛问题的完整排查记录 |
| [09-ppu](09-ppu.md) | PPU ZW810E | CLIP 已重训复核 |
| [16-ascend](16-ascend.md) | 昇腾 910C | GEMM 比 A100 快，慢在小算子 |
| [14-inference-throughput](14-inference-throughput.md) | 四平台 | 推理口径的**方法论**（两套系数推导、样本档位差异）。**逐平台的训练+推理并排表在各芯片文档的「吞吐总表」里** |

## data/ — 原始产物

```
data/inference/{a100,kunlunxin,ascend,ppu}_{bge,clip,rr_base}.json
data/op-micro/ascend_910c.json
data/ppu_clip_{base,trained}.json
```

每份 JSON 都含 `inference` 段（`items_scored` / `model_seconds` / `samples_per_s`），
文档里的每个吞吐数字都能追溯到具体文件。

## 两个口径不要混

- **训练吞吐**（07-09/16）：含反向 + 优化器，run 全程折算
- **推理吞吐**（14）：只计模型前向

同平台同模型二者差 3-4 倍（A100 BGE：训练 118 → 推理 395，3.35×）。
