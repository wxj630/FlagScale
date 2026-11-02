"""Is the reranker validation loss rising because of overfitting or a data bug?

The A100 run shows train 0.180 -> 0.046 while val 0.721 -> 1.036.  Before
blaming the optimiser, this checks the things that would make val look worse
without the model being wrong:

  * are val queries really disjoint from train queries?
  * how many queries and rows are in each split?
  * what does each label's score distribution look like?
  * does the base (untrained) checkpoint score better than the trained one?

    python diagnose_reranker_val.py --conf examples/retrieval/conf/train/qwen3_reranker.yaml
"""
import argparse, collections, json
import torch
from omegaconf import OmegaConf

ap = argparse.ArgumentParser()
ap.add_argument("--conf", required=True)
ap.add_argument("--rows", type=int, default=4000, help="rows to profile per split")
ap.add_argument("--checkpoints", nargs="*", default=[],
                help="extra checkpoints to score on the same val rows")
a = ap.parse_args()

from flagscale.train.datasets.retrieval import build_dataset
from flagscale.train.losses.retrieval import reranker_batch

cfg = OmegaConf.load(a.conf)
data = cfg.data
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def collect(split, limit):
    ds = build_dataset(str(cfg.task), data, str(data.path), split, None)
    rows, queries = [], set()
    for row in ds:
        rows.append(row)
        queries.add(str(row.get("anchor", "")))
        if len(rows) >= limit:
            break
    return rows, queries


print("=== split composition ===")
tr_rows, tr_q = collect("train", a.rows)
va_rows, va_q = collect("validation", a.rows)
print(f"train: {len(tr_rows)} rows, {len(tr_q)} unique queries")
print(f"val  : {len(va_rows)} rows, {len(va_q)} unique queries")
if tr_q and va_q:
    overlap = tr_q & va_q
    print(f"query overlap: {len(overlap)} ({len(overlap)/len(va_q)*100:.1f}% of val queries)")
print(f"rows per query: train {len(tr_rows)/max(1,len(tr_q)):.1f}, "
      f"val {len(va_rows)/max(1,len(va_q)):.1f}")

print()
print("=== row fields ===")
print(sorted(va_rows[0].keys()))
sample = va_rows[0]
for k, v in sample.items():
    print(f"  {k}: {str(v)[:110]}")

# How different are the negatives?  Identical negative text reused across rows
# would mean the model can memorise rather than learn relevance.
neg_keys = sorted(k for k in va_rows[0] if k.startswith("negative"))
for k in neg_keys:
    texts = [str(r.get(k, "")) for r in va_rows]
    uniq = len(set(texts))
    print(f"{k}: {len(texts)} rows, {uniq} unique ({uniq/max(1,len(texts))*100:.0f}%)")


def score_loss(rows, model_path=None):
    """BCE loss of one checkpoint on the same rows."""
    from flagscale.models.retrieval import build_retrieval_model
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
    total, count = 0.0, 0
    bs = 32
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            chunk = pairs[i:i + bs]
            scores = reranker_batch(model, chunk, dev)
            y = torch.tensor(labels[i:i + bs], device=dev)
            total += float(torch.nn.functional.binary_cross_entropy_with_logits(scores, y))
            count += 1
    return total / max(1, count)


val_rows = va_rows[:a.rows]
print()
print(f"=== BCE loss on {len(val_rows)} validation rows ===")
base_loss = score_loss(val_rows)
print(f"  base (pretrained)          {base_loss:.4f}")
for ckpt in a.checkpoints:
    print(f"  {ckpt.split('/')[-1]:26} {score_loss(val_rows, ckpt):.4f}")
