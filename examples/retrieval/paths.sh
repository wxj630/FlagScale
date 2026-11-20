#!/bin/bash
# Path roots for every Kunlunxin script.  SOURCE this, do not execute it.
#
# Why a shared file: the P800 hosts moved to a Lustre volume on 2026-11-20 so
# both machines read one copy of models / datasets / checkpoints / envs / code.
# Hard-coding /data2/jiangxuhui in five scripts meant the second host could not
# start anything at all -- its /data2/jiangxuhui does not exist. One root
# variable makes the machines interchangeable and keeps a future move to one
# line.
#
# FLAGOS_DATA_ROOT     data + envs + checkpoints (Lustre, shared by both hosts)
# FLAGOS_CODE_ROOT     where FlagScale / Torch-FL / megatron-reranker live
#
# /flagos-retrieval-{models,datasets,ckpts} are symlinks into FLAGOS_DATA_ROOT, so
# config files keep using the stable /flagos-retrieval-* paths (AGENTS.md s.9)
# and do not care which host or which disk is behind them.

export FLAGOS_DATA_ROOT="${FLAGOS_DATA_ROOT:-/data_427/jiangxuhui}"
export FLAGOS_CODE_ROOT="${FLAGOS_CODE_ROOT:-/data_427/jiangxuhui/code}"

export FLAGOS_MODELS="${FLAGOS_MODELS:-$FLAGOS_DATA_ROOT/models}"
export FLAGOS_DATASETS="${FLAGOS_DATASETS:-$FLAGOS_DATA_ROOT/datasets}"
export FLAGOS_CKPTS="${FLAGOS_CKPTS:-$FLAGOS_DATA_ROOT/ckpts}"
export FLAGOS_ENVS="${FLAGOS_ENVS:-$FLAGOS_DATA_ROOT/envs}"
export FLAGOS_MEGATRON="${FLAGOS_MEGATRON:-$FLAGOS_DATA_ROOT/megatron-reranker}"

# Scratch. Must not be under / : it is ~98% full and ENOSPC kills runs mid-way.
export FLAGOS_SCRATCH="${FLAGOS_SCRATCH:-$FLAGOS_DATA_ROOT/scratch}"

_vendor_env="$FLAGOS_DATA_ROOT/flagos-retrieval-vendor-env.sh"
if [ -f "$_vendor_env" ]; then
    # shellcheck disable=SC1090
    source "$_vendor_env"
fi
unset _vendor_env
