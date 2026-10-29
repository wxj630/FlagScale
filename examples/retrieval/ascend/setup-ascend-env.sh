#!/usr/bin/env bash
# Build an isolated Ascend (torch_npu) environment for FlagScale retrieval.
# Safe to re-run: every step is idempotent.
set -euo pipefail

PROJECT_ROOT="${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
FLAGSCALE_DIR="$PROJECT_ROOT/FlagScale"
VENV="${VENV:-/home/jiangxuhui/venvs/flagos-ascend}"
UV="${UV:-$HOME/.local/bin/uv}"
PY_VERSION="${PY_VERSION:-3.11.14}"
PY_RELEASE="${PY_RELEASE:-20251014}"
PYTHON_HOME="${PYTHON_HOME:-/home/jiangxuhui/venvs/python-${PY_VERSION}}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
# GitHub is very slow/unstable from this host; use a release mirror for the
# CPython standalone tarball. Override with "" to use GitHub directly.
PY_MIRROR="${PY_MIRROR:-https://ghfast.top/https://github.com/astral-sh/python-build-standalone/releases/download}"
PY_TARBALL="cpython-${PY_VERSION}+${PY_RELEASE}-aarch64-unknown-linux-gnu-install_only.tar.gz"
PY_CACHE="${PY_CACHE:-$HOME/.cache/flagos-python}"
export UV_DEFAULT_INDEX="$PIP_INDEX"
export UV_INDEX_URL="$PIP_INDEX"

mkdir -p "$PROJECT_ROOT/logs"
LOG="$PROJECT_ROOT/logs/setup-ascend-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "===== setup-ascend-env start: $(date '+%F %T') ====="
echo "project: $PROJECT_ROOT"
echo "venv:    $VENV"
echo "log:     $LOG"

# CANN runtime (libhccl.so etc.)
# set_env.sh reads possibly-unset vars; relax nounset while sourcing it.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"

# 1) Python >=3.10 (FlagScale requirement), from a standalone build.
#    GitHub/uv downloads stall on this host, so use an explicit mirror + retries.
if [ ! -x "$PYTHON_HOME/bin/python3.11" ]; then
  mkdir -p "$PY_CACHE" "$(dirname "$PYTHON_HOME")"
  if [ ! -s "$PY_CACHE/$PY_TARBALL" ]; then
    echo "downloading $PY_TARBALL"
    curl -fL --retry 8 --retry-delay 5 --connect-timeout 30 \
      "$PY_MIRROR/$PY_RELEASE/$PY_TARBALL" -o "$PY_CACHE/$PY_TARBALL"
  fi
  tmpdir="$(mktemp -d)"
  tar -xzf "$PY_CACHE/$PY_TARBALL" -C "$tmpdir"
  rm -rf "$PYTHON_HOME"
  mv "$tmpdir/python" "$PYTHON_HOME"
  rmdir "$tmpdir"
fi
echo "python home: $PYTHON_HOME"
"$PYTHON_HOME/bin/python3.11" -V

# 2) isolated venv (never touch the adrec training venv)
if [ ! -x "$VENV/bin/python" ]; then
  "$UV" venv --seed --python "$PYTHON_HOME/bin/python3.11" "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
  || { echo "ERROR: $PY is not >=3.10"; exit 1; }
echo "python: $("$PY" -V)"

# 3) vendor runtime: must match the host CANN 8.5.0 stack
"$UV" pip install --python "$PY" --index-url "$PIP_INDEX" \
  'torch==2.6.0' 'torchvision==0.21.0' 'torch_npu==2.6.0.post5'

# 4) FlagScale retrieval dependencies (Ascend base + train profile)
"$UV" pip install --python "$PY" --index-url "$PIP_INDEX" \
  -r "$FLAGSCALE_DIR/requirements/ascend/train.txt"

# datasets resolves the latest pyarrow, which now requires NumPy>=2, while
# FlagScale pins NumPy 1.26.4. pyarrow 21 still supports NumPy 1.x and is the
# minimum accepted by datasets 4.5.0.
"$UV" pip install --python "$PY" --index-url "$PIP_INDEX" 'pyarrow==21.0.0'

# 5) FlagScale itself, editable; deps are already installed above
"$UV" pip install --python "$PY" --no-deps -e "$FLAGSCALE_DIR"

# 6) verify imports / device visibility
"$PY" - <<'PYEOF'
import torch
import torch_npu
import flagscale
print("torch       :", torch.__version__)
print("torch_npu   :", torch_npu.__version__)
print("npu_available:", torch.npu.is_available())
print("npu_count   :", torch.npu.device_count())
print("flagscale   :", flagscale.__file__)
PYEOF

echo "===== setup-ascend-env done: $(date '+%F %T') ====="
touch "$PROJECT_ROOT/logs/.setup-ascend-ok"
