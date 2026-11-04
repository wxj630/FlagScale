#!/bin/bash
# Poll the three reranker runs and report progress + validation trend.
#
# Why: the shared recipe is 3 epochs at lr 1e-5, but the 2026-10-27 A100 run
# showed validation loss rising after epoch 1 (0.721 -> 1.036), i.e. epoch 1 was
# already the best.  Spending another two days past that point buys nothing.
# This script only observes; the decision to stop is a human one.
set -uo pipefail
ENVF=/Volumes/PSSD/code/FlagOS/FlagScale/.env
PW=$(sed -n 's/^PPU_PASSWORD=//p' "$ENVF" | tail -1)

a100() {
  ssh -o ConnectTimeout=20 -o BatchMode=yes bitahub_2gpus '
    L=/flagos-search-ckpts/a100-fp32master-3epoch/qwen3_reranker/logs/train-qwen3_reranker.log
    S=$(grep -a "optimizer_step=" "$L" | tail -1 | sed -E "s/.*optimizer_step=([0-9]+).*/\1/")
    E=$(grep -a "train_loss=" "$L" | tail -3 | tr "\n" " ")
    echo "A100 step=${S:-?}/4783 per-epoch; epochs: ${E:-none yet}"' 2>&1 | tail -2
}

kunlunxin() {
  ssh -o ConnectTimeout=20 -o BatchMode=yes kunlunxin '
    L=/data2/jiangxuhui/s640full.log
    S=$(tail -1 "$L" | grep -aoE "step=[0-9]+" | head -1)
    R=$(tail -1 "$L" | grep -aoE "pairs/s=[0-9.]+" | head -1)
    END=$(grep -a "train_loss=" "$L" | tail -1)
    if pgrep -f "train_retrieval.py" >/dev/null 2>&1; then ST=running; else ST=finished; fi
    echo "KLX ${S:-?} ${R:-?} $ST; ${END:-no epoch end yet}"' 2>&1 | tail -2
}

ppu() {
  printf "%s\n" "$PW" | ssh -o ConnectTimeout=20 -o BatchMode=yes pingtouge '
    read -r p
    printf "%s\n" "$p" | sudo -S -p "" docker exec flagos-retrieval bash -c "
      L=/workspace/checkpoints/ppu-fp32master-3epoch/qwen3_reranker/logs/train-qwen3_reranker.log
      S=\$(grep -a optimizer_step= \$L | tail -1 | sed -E \"s/.*optimizer_step=([0-9]+).*/\1/\")
      E=\$(grep -a train_loss= \$L | tail -3 | tr \"\n\" \" \")
      echo \"PPU step=\${S:-?}/4783; epochs: \${E:-none yet}\"
    " 2>&1 | tail -1' 2>&1 | tail -1
}

echo "=== $(date '+%F %T') ==="
a100
kunlunxin
ppu
