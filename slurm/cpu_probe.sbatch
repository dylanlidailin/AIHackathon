#!/bin/bash
#SBATCH -A ai-safety-hackathon-rig
#SBATCH -p standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J xmprobe-cpu
#SBATCH -o logs/%x-%j.out
# Usage: sbatch slurm/cpu_probe.sbatch <config>
# If the hackathon account has no access to `standard`, submit with `-p gpu` instead
# (the probe stage is CPU-only either way; see docs/quickstart.md step 8).
set -euo pipefail
CONFIG=${1:?usage: sbatch slurm/cpu_probe.sbatch <config>}

module load miniforge
eval "$(conda shell.bash hook)"
conda activate xmprobe
cd "$SLURM_SUBMIT_DIR"

python -m xmprobe.cli probe  --config "$CONFIG"
python -m xmprobe.cli report --config "$CONFIG"
