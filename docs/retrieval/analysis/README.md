# analysis/ — 分析

支撑结论的推导：算力换算、折扣系数、慢算子归因。

| 文档 | 用途 |
|---|---|
| [10-microbench-discount](10-microbench-discount.md) | **折扣系数**——达标判定的核心。六家芯片的 bf16/FP16/带宽实测，两种口径（含/不含 FP32） |
| [13-bottleneck-ops](13-bottleneck-ops.md) | **慢算子清单**，按"占差距的比例"排序，交给算子适配方 |
| [06-gpu-hours](06-gpu-hours.md) | GPU-hours 估算（三口径），含论文自己声明的集群 |
| [11-vendor-questions](11-vendor-questions.md) | 给厂商的问题清单 |

> ⚠️ **折扣系数不是验收方给的数值。** docx §4.2 只给了算法（几何平均）和
> `× 0.8`，系数部分原文标为"建议讨论"。我们算出的 1.321 / 1.050 / 0.711
> 是推导值，需对方确认——见 [10](10-microbench-discount.md) §一·五。
