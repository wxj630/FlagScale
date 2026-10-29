#!/usr/bin/env bash
# Source this before launching FlagScale retrieval jobs on Ascend:
#   source scripts/ascend-env.sh
#   export ASCEND_RT_VISIBLE_DEVICES=0   # pick an idle chip (0-15)

PROJECT_ROOT="${FLAGSCALE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
VENV="${VENV:-/home/jiangxuhui/venvs/flagos-ascend}"

case $- in *u*) _restore_u=1 ;; *) _restore_u=0 ;; esac
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source "$VENV/bin/activate"
if [ "$_restore_u" = 1 ]; then set -u; fi
unset _restore_u

export FS_PLATFORM="${FS_PLATFORM:-npu}"
export FLAGSCALE_DEVICE="${FLAGSCALE_DEVICE:-npu}"
export FLAGSCALE_PYTHON_ENV="${FLAGSCALE_PYTHON_ENV:-$VENV}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$PROJECT_ROOT/FlagScale:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
