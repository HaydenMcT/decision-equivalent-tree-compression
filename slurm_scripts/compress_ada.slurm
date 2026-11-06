#!/bin/bash
#SBATCH --job-name=ada
#SBATCH -c 1
#SBATCH -t 40:00
#SBATCH -o slurm_logs/%j.out
#SBATCH -e slurm_logs/%j.err
#SBATCH --mem=50G
#SBATCH --partition=compsci
#SBATCH --array=24-35
#SBATCH --nodelist=linux[31-40]
#SBATCH --nodes=1


# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline
python -m experiments.count_full_de_compression_adaboost --index $((SLURM_ARRAY_TASK_ID)) --rerun