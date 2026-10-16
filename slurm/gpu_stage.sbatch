#!/bin/bash
#SBATCH -A ai-safety-hackathon-rig
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -J xmprobe-gpu
#SBATCH -o logs/%x-%j.out
# Usage: sbatch slurm/gpu_stage.sbatch <config> <stage> [limit]
#   stage: render | ocr_check | generate | extract | demo | all
set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/gpu_stage.sbatch <config> <stage> [limit]}
STAGE=${2:?stage}
LIMIT=${3:-}

module load miniforge
eval "$(conda shell.bash hook)"
conda activate xmprobe
export HF_HOME=/scratch/esd4uq/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

nvidia-smi --query-gpu=name,memory.total --format=csv
python -m xmprobe.cli "$STAGE" --config "$CONFIG" ${LIMIT:+--limit "$LIMIT"}
