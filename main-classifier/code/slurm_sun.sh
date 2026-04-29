#!/bin/bash
# ============================================================
# SUN Attributes Training — SLURM job script
#
# Usage:
#   sbatch slurm_sun.sh
#
# Monitor your job:
#   myq                      # check job status
#   cat slurm-<jobid>.out    # view stdout
#   cat slurm-<jobid>.err    # view stderr
# ============================================================

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -J sun_attr_train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

PROJECT_DIR=~/sun-project/main-classifier/code

source module load python/3.11 cuda/12.9.0-cinr && source ~/envs/csci1430/bin/activate

cd "$PROJECT_DIR"

# Run training
python train_sun.py

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"
