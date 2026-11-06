#!/bin/bash
#SBATCH --job-name=tune
#SBATCH -c 1
#SBATCH -t 2:00:00
#SBATCH -o slurm_logs/equiv_tune_%j.out
#SBATCH -e slurm_logs/equiv_tune_%j.err
#SBATCH --mem=20G
#SBATCH --partition=compsci
#SBATCH --array=12-23,36-47

# RUN THIS FILE FROM THE BASE DIRECTORY

# Turn off numpy's AVX-512 paths so that tuned results are comparable to those on the hardware used for Adaboost compression timing
export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"

# run baseline
python -m experiments.tune_models --index $((SLURM_ARRAY_TASK_ID)) --rerun