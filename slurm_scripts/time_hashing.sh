#!/bin/bash
#SBATCH --job-name=hashing
#SBATCH -t 24:00:00
#SBATCH -o slurm_logs/%j_%a.out
#SBATCH -e slurm_logs/%j_%a.err
#SBATCH --array=1-48,97-144,193-240,289-336
#SBATCH --partition=compsci
#SBATCH --mem=100G
#SBATCH --nodelist=compsci-cluster-fitz-[35-44]
#SBATCH --nodes=1

# RUN THIS FILE FROM THE BASE DIRECTORY

# run baseline 
python -m experiments.rset_compress --index $((SLURM_ARRAY_TASK_ID)) --rerun
