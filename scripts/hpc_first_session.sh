#!/bin/bash
# First GPU session (design D9 steps 1, 3, 3b): toolchain check, smoke run, demo.
# Usage (inside a GPU allocation, from the repo root): bash scripts/hpc_first_session.sh [smoke|demo|all]
#   smoke: toolchain check + smoke run on Qwen3.5-0.8B
#   demo:  toolchain check + 100-pair demo on Qwen3.5-2B
# Submitted via slurm/session.sbatch. Every step runs even if an earlier one fails; a summary is
# printed at the end.
set -uo pipefail
MODE=${1:-all}
module load miniforge
eval "$(conda shell.bash hook)"
conda activate xmprobe
# `conda activate` can silently leave another python first on PATH in batch shells; pin the env
# explicitly and ignore ~/.local user site-packages.
export PATH=$HOME/.conda/envs/xmprobe/bin:$PATH PYTHONNOUSERSITE=1
python -c 'import sys; assert sys.prefix.endswith("xmprobe"), sys.prefix' || exit 1
export HF_HOME=/scratch/esd4uq/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
SMALL=configs/qwen35_0p8b.yaml
BIG=configs/qwen35_2b.yaml
declare -a SUMMARY

step() {  # step <name> <command...>
  local name=$1; shift
  echo; echo "===== [$name] $(date +%T) $*"
  local t0=$SECONDS
  "$@"
  local rc=$?
  SUMMARY+=("$name rc=$rc $((SECONDS - t0))s")
  echo "===== [$name] rc=$rc in $((SECONDS - t0))s"
  return $rc
}

nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"


# Smoke: `all` stops at the OCR gate with 10 pairs (n < 30 by design); then finish stage by stage.
smoke() {
  python -m xmprobe.cli all --config $SMALL --limit 10
  local rc=$?
  [ $rc -eq 0 ] || [ $rc -eq 3 ] || return $rc
  for s in generate extract probe report; do
    python -m xmprobe.cli $s --config $SMALL --limit 10 || return $?
  done
}
if [ "$MODE" = smoke ] || [ "$MODE" = all ]; then
  step toolchain_0p8b python scripts/check_toolchain.py $SMALL
  step smoke_0p8b smoke
fi
if [ "$MODE" = demo ] || [ "$MODE" = all ]; then
  step toolchain_2b python scripts/check_toolchain.py $BIG
  step demo_2b python -m xmprobe.cli demo --config $BIG
fi

echo; echo "===== SUMMARY"
printf '%s\n' "${SUMMARY[@]}"
echo SESSION_DONE
