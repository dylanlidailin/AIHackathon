#!/bin/bash
#SBATCH -A ai-safety-hackathon-rig
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
# Usage: sbatch -J xmprobe-smoke slurm/session.sbatch smoke
#        sbatch -J xmprobe-demo  slurm/session.sbatch demo
# Override the account on the command line (e.g. -A aikyamlab) while the hackathon account is unavailable.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
bash scripts/hpc_first_session.sh "${1:?usage: sbatch slurm/session.sbatch <smoke|demo|all>}"
