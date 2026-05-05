#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 10:00:00
#SBATCH -J geoguessr
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Default task if none provided as argument
TASK=${1:-t0_attention}

echo "============================================"
echo "Job ID:    $SLURM_JOB_ID"
echo "Task:      $TASK"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "============================================"

# Run from the code directory
cd "$SLURM_SUBMIT_DIR"

source ~/.local/bin/env

# Run training
uv run python main.py --task "$TASK"

echo "============================================"
echo "Finished:  $(date)"
echo "============================================"