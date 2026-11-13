#!/usr/bin/env bash
# Download the continued-pretraining corpora from ModelScope.
#
# Every dataset here was verified to exist through the ModelScope API (which
# returns an explicit "不存在的数据集" for a missing repo) and every field name was
# checked against the published dataset_infos.json.  Sizes are the repository
# totals, not an estimate.
#
#   BGE-M3      C-MTEB/LCQMC                              14 MB   sentence1/sentence2/score
#   CLIP        lmms-lab/Flickr30K                      4.4 GB   image/caption
#   Reranker    mteb/MSMARCO                            3.5 GB   corpus/queries/qrels
#
# Two of the three need a conversion step before the trainer can read them:
#   * Flickr30K is parquet already -> `data.format: parquet`, no conversion.
#   * MS MARCO is three tables -> `tools/mteb_to_triplets.py`.
#   * LCQMC is parquet with sentence1/sentence2 -> `data.format: pair`, no
#     conversion (the reader renames the columns and drops score==0 rows).
#
# Usage:
#   bash examples/retrieval/download-continual-pretrain.sh            # all three
#   bash examples/retrieval/download-continual-pretrain.sh lcqmc      # one
set -euo pipefail

ROOT=${FLAGSCALE_RETRIEVAL_DATASETS:-/flagos-retrieval-datasets}
PY=${FLAGSCALE_PYTHON:-python}

# ModelScope needs the `modelscope` package on every machine that downloads.
# Its install is the one thing here that touches the public index, and one box
# (Kunlunxin) timed out on files.pythonhosted.org, so allow a mirror override:
#   PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ bash <this script>
ensure_modelscope() {
  if $PY -c 'import modelscope' >/dev/null 2>&1; then
    return
  fi
  echo "==> installing modelscope"
  $PY -m pip install --no-input --timeout 120 \
      ${PIP_INDEX_URL:+-i "$PIP_INDEX_URL"} modelscope
}

fetch() {  # fetch <modelscope-id> <local-name>
  local repo="$1" dest="$ROOT/$2"
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    echo "==> skip $2 (already present at $dest)"
    return
  fi
  echo "==> $repo -> $dest"
  mkdir -p "$dest"
  $PY - "$repo" "$dest" <<'PYEOF'
import sys

from modelscope import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
# `repo_type="dataset"` is required: the default looks the id up in the *model*
# index and fails with a 404 "record not found" even though the dataset exists.
path = snapshot_download(repo, repo_type="dataset", local_dir=dest)
print(f"    downloaded to {path}")
PYEOF
}

want=${1:-all}

ensure_modelscope

if [ "$want" = "all" ] || [ "$want" = "lcqmc" ]; then
  fetch C-MTEB/LCQMC lcqmc
fi

if [ "$want" = "all" ] || [ "$want" = "flickr30k" ]; then
  fetch lmms-lab/Flickr30K flickr30k
fi

if [ "$want" = "all" ] || [ "$want" = "msmarco" ]; then
  fetch mteb/MSMARCO msmarco
  echo "==> converting MS MARCO qrels -> reranker triplets"
  $PY "$(dirname "$0")/../../tools/mteb_to_triplets.py" \
      --dataset-dir "$ROOT/msmarco" \
      --out "$ROOT/msmarco/triplets" \
      --num-negatives 1 --max-queries 20000 --seed 42
fi

cat <<'EOF'

Done. Run one of:

  python -m flagscale.run --config-path examples/retrieval/conf --config-name train \
    action=test train=bge_m3_continual_lcqmc train.parallel=ddp \
    experiment.runner.nproc_per_node=1 train.hardware.device_type=cuda \
    train.output_dir=... experiment.exp_dir=...

  python -m flagscale.run --config-path examples/retrieval/conf --config-name train \
    action=test train=clip_vit_large_flickr30k train.parallel=ddp \
    experiment.runner.nproc_per_node=1 train.hardware.device_type=cuda \
    train.output_dir=... experiment.exp_dir=...

  python -m flagscale.run --config-path examples/retrieval/conf --config-name train \
    action=test train=qwen3_reranker_msmarco train.parallel=fsdp2 \
    experiment.runner.nproc_per_node=2 train.hardware.device_type=cuda \
    train.output_dir=... experiment.exp_dir=...
EOF
