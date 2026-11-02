"""Where does the fine-tuned reranker actually lose to the pretrained one?

The repaired checkpoint scores 0.618 BCE against the base model's 0.429 on the
same rows -- fine-tuning made it worse, and the earlier reload bug had hidden
that.  BCE alone cannot say how to fix it, so this splits the score distribution
by label and reports accuracy and the margin between positives and negatives.

A model that collapses both scores toward a constant looks very different from
one that overfits positives, and the fix differs too.
"""
import argparse, statistics
import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--conf", required=True)
ap.add_argument("--rows", type=int, default=800)
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
print(f"validation rows: {len(rows)}, negatives per row: {len(neg_keys)}")
print(f"queries: {len({r['anchor'] for r in rows})}")


def profile(model_path=None):
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
            s = reranker_batch(model, pairs[i:i + 32], dev)
            scores.extend(float(x) for x in s)
    pos = [s for s, y in zip(scores, labels) if y > 0.5]
    neg = [s for s, y in zip(scores, labels) if y <= 0.5]

    # Pairwise accuracy: does a positive outrank the negatives of its own query?
    correct = total = 0
    per_row = len(neg_keys) + 1
    for i in range(len(rows)):
        p = scores[i * per_row]
        for j in range(1, per_row):
            n = scores[i * per_row + j]
            total += 1
            correct += int(p > n)

    return {
        "pos_mean": statistics.mean(pos), "pos_std": statistics.pstdev(pos),
        "neg_mean": statistics.mean(neg), "neg_std": statistics.pstdev(neg),
        "margin": statistics.mean(pos) - statistics.mean(neg),
        "pairwise_acc": correct / max(1, total),
        "pos_min": min(pos), "pos_max": max(pos),
    }


print()
print(f"{'model':22}{'pos mean':>10}{'neg mean':>10}{'margin':>9}{'pos std':>9}{'pair acc':>10}")
base = profile()
print(f"{'base':22}{base['pos_mean']:>10.3f}{base['neg_mean']:>10.3f}"
      f"{base['margin']:>9.3f}{base['pos_std']:>9.3f}{base['pairwise_acc']:>10.1%}")
for ckpt in a.checkpoints:
    m = profile(ckpt)
    name = ckpt.rstrip("/").split("/")[-1][:20]
    print(f"{name:22}{m['pos_mean']:>10.3f}{m['neg_mean']:>10.3f}"
          f"{m['margin']:>9.3f}{m['pos_std']:>9.3f}{m['pairwise_acc']:>10.1%}")

print()
print("how to read this:")
print("  margin    = pos_mean - neg_mean; bigger is better, negative means inverted")
print("  pos std   = spread of positive scores; near zero means the model collapsed")
print("  pair acc  = fraction of (positive, its own negatives) pairs ranked correctly")
