#!/bin/bash
#SBATCH --job-name=rsetcount
#SBATCH -c 1
#SBATCH -t 8:00:00
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err
#SBATCH --mem=50G
#SBATCH --array=12,13,14,16

# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline 
python -m experiments.count_rset --index $((SLURM_ARRAY_TASK_ID)) --rerun
python -m experiments.continuous_count_rset --index $((SLURM_ARRAY_TASK_ID)) --rerun
