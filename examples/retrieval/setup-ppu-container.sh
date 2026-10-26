#!/bin/bash
# Prepare a PPU container to run the retrieval experiments.
#
# Everything a fresh container needs, in one place, so a rebuilt container is
# reproducible instead of depending on what someone typed earlier:
#
#   1. The /flagos-retrieval-* symlinks the shared configs point at.
#   2. Python packages the base image lacks (hydra, flag_gems, ...).
#
# The symlinks exist because the configs are deliberately identical across
# machines (AGENTS.md section 8): they name /flagos-retrieval-{models,datasets,
# ckpts} and each machine resolves those to its own storage. On PPU the host
# / is read-only, so the links have to live inside the container.
#
#   bash /workspace/setup-ppu-container.sh [src-site-packages]
#
# src-site-packages is a container that already has the packages installed; its
# site-packages files are copied in. Omit it when the image is already complete.
set -uo pipefail

SRC=${1:-}
SP=/usr/local/lib/python3.12/site-packages
WS=/workspace

echo "==> symlinks"
ln -sfn "$WS/models" /flagos-retrieval-models
ln -sfn "$WS/data" /flagos-retrieval-datasets
mkdir -p /flagos-retrieval-ckpts
ln -sfn "$WS/checkpoints" /flagos-retrieval-ckpts/store

# The shared configs use a two-level layout (<name>/<name>), while the PPU copy
# of the data is flat. Link the inner level rather than editing the configs.
for m in bge-m3 clip-vit-large-patch14 Qwen3-Reranker-8B; do
  [ -d "$WS/models/$m" ] && ln -sfn "$WS/models/$m" "$WS/models/$m/$m"
done
for d in t2ranking wds_mscoco_captions2017; do
  [ -d "$WS/data/$d" ] && ln -sfn "$WS/data/$d" "$WS/data/$d/$d"
done

for p in "$WS/models/bge-m3/bge-m3" \
         "$WS/models/clip-vit-large-patch14/clip-vit-large-patch14" \
         "$WS/models/Qwen3-Reranker-8B/Qwen3-Reranker-8B" \
         "$WS/data/t2ranking/t2ranking/triplet-15" \
         "$WS/data/wds_mscoco_captions2017/wds_mscoco_captions2017/train" \
         "$WS/data/wds_mscoco_captions2017/wds_mscoco_captions2017/test"; do
  if [ -e "$p" ]; then
    echo "  ok   $p"
  else
    echo "  MISS $p" >&2
  fi
done

if [ -n "$SRC" ]; then
  echo "==> copying packages from $SRC"
  # Only the packages this image lacks; copying everything would clobber the
  # vendor torch.
  comm -23 \
    <(docker exec "$SRC" bash -c "ls $SP" | sort) \
    <(ls "$SP" | sort) > /tmp/missing.txt
  wc -l < /tmp/missing.txt | xargs echo "  packages to copy:"
  docker cp /tmp/missing.txt "$SRC":/tmp/missing.txt 2>/dev/null || true
  : # the copy itself is driven from the host; see run-ppu-setup.sh
fi

echo "==> environment check"
cd "$WS/FlagScale" || exit 1
PYTHONPATH="$WS/Torch-FL:$WS/FlagScale" FLAGOS_USE_FLAGGEMS=false GEMS_VENDOR=thead \
python - <<'PY'
import torch_fl, torch
import hydra  # noqa: F401
print(f"  torch {torch.__version__}, devices {torch.cuda.device_count()}")
PY

echo "done."
