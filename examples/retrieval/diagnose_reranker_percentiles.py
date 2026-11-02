"""Percentile breakdown: why is BCE worse when the margin got bigger?

epoch-1 has a larger positive/negative margin than the base model (5.29 vs
3.05) yet a worse BCE (0.618 vs 0.429).  Averaging hides the answer: BCE is
dominated by the worst examples.  This prints the score distribution by
percentile so a collapsed tail is visible.
"""
import argparse
import statistics
import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--conf", required=True)
ap.add_argument("--rows", type=int, default=600)
ap.add_argument("--checkpoints", nargs="*", default=[])
a = ap.parse_args()

from flagscale.train.datasets.retrieval import build_dataset
from flagscale.models.retrieval import build_retrieval_model
from flagscale.train.losses.retrieval import reranker_batch

cfg = OmegaConf.load(a.conf)
data = cfg.data
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
ds = build_dataset(str(cfg.task), data, str(data.validation_path), "validation", a.rows)
rows = list(ds)
neg_keys = sorted(k for k in rows[0] if k.startswith("negative"))


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * q))]


def collect(model_path=None):
    mcfg = OmegaConf.create(dict(cfg.model))
    if model_path:
        mcfg.model_path = model_path
    model, processor = build_retrieval_model(str(cfg.model.model_type), mcfg, dev)
    model.eval()
    pairs, labels = [], []
    for r in rows:
        pairs.append([str(r["anchor"]), str(r["positive"])])
        labels.append(1.0)
        for k in neg_keys:
            pairs.append([str(r["anchor"]), str(r[k])])
            labels.append(0.0)
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), 32):
            scores.extend(float(x) for x in reranker_batch(model, pairs[i:i + 32], dev))
    return scores, labels


def report(name, scores, labels):
    pos = [s for s, y in zip(scores, labels) if y > 0.5]
    neg = [s for s, y in zip(scores, labels) if y <= 0.5]
    print(f"\n{name}")
    print(f"  positives n={len(pos)}")
    print(f"    p5 {pct(pos,0.05):7.2f}  p25 {pct(pos,0.25):7.2f}  "
          f"med {pct(pos,0.50):7.2f}  p75 {pct(pos,0.75):7.2f}  p95 {pct(pos,0.95):7.2f}")
    print(f"    min {min(pos):7.2f}  max {max(pos):7.2f}")
    print(f"  negatives n={len(neg)}")
    print(f"    p5 {pct(neg,0.05):7.2f}  p25 {pct(neg,0.25):7.2f}  "
          f"med {pct(neg,0.50):7.2f}  p75 {pct(neg,0.75):7.2f}  p95 {pct(neg,0.95):7.2f}")

    # The loss is the sum of these two failure masses.
    pos_wrong = [s for s in pos if s < 0]
    neg_wrong = [s for s in neg if s > 0]
    p = torch.sigmoid(torch.tensor(pos))
    n = torch.sigmoid(torch.tensor(neg))
    bce = float(-(torch.log(p).mean() + torch.log(1 - n).mean()) / 2)
    print(f"  BCE {bce:.4f}   positives scored <0: {len(pos_wrong)} "
          f"({len(pos_wrong)/len(pos)*100:.1f}%)   negatives scored >0: {len(neg_wrong)} "
          f"({len(neg_wrong)/len(neg)*100:.1f}%)")

    # How much of the total loss comes from the worst 5% of positives?
    worst = sorted(pos)[:max(1, len(pos) // 20)]
    worst_contrib = float(-torch.log(torch.sigmoid(torch.tensor(worst))).sum()) / len(pos)
    print(f"  worst 5% of positives contribute {worst_contrib:.4f} of the {bce:.4f} BCE")


report("base", *collect())
for ckpt in a.checkpoints:
    report(ckpt.rstrip("/").split("/")[-1], *collect(ckpt))
