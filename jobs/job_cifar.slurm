#!/bin/bash
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 600 # Runtime in minutes (10 hours)
#SBATCH -p seas_gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=80000 # Memory total in MB
#SBATCH --open-mode=append
#SBATCH -o /n/home12/ishapira/git_connection/TinyML-RegulizationAndQuantization/jobs/cifar_slurm_logs/cifar0005_{TYPE}_{PARAM}_%j.out # Standard out goes to this file
#SBATCH -e /n/home12/ishapira/git_connection/TinyML-RegulizationAndQuantization/jobs/cifar_slurm_logs/cifar0005_{TYPE}_{PARAM}_%j.err # Standard err goes to this file

module load Anaconda2/2019.10-fasrc01
source activate itai_ml_env

# Navigate to the directory where the script is located
cd /n/home12/ishapira/git_connection/TinyML-RegulizationAndQuantization/jobs

# Execute the Python script
python3 CIFAR_job.py --regularization_type {TYPE} --regularization_param {PARAM}

# Save GPU info
nvidia-smi > /n/home12/ishapira/git_connection/TinyML-RegulizationAndQuantization/jobs/gpu.txt

# Deactivate Conda environment
conda deactivate