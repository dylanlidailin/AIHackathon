#!/bin/bash
#SBATCH -A ai-safety-hackathon-rig
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J xmprobe-toolchain
#SBATCH -o logs/%x-%j.out
# Usage: sbatch slurm/toolchain.sbatch <config>   (exits non-zero unless GATE: PASS)
set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/toolchain.sbatch <config>}
module load miniforge
eval "$(conda shell.bash hook)"
conda activate xmprobe
export PATH=$HOME/.conda/envs/xmprobe/bin:$PATH PYTHONNOUSERSITE=1
python -c 'import sys; assert sys.prefix.endswith("xmprobe"), sys.prefix'
export HF_HOME=/scratch/esd4uq/hf_cache HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv
python scripts/check_toolchain.py "$CONFIG"
