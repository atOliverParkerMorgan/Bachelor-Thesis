#!/bin/bash
#SBATCH --partition=arm_fast
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --job-name=nnunet-train-arm
#SBATCH --output=slurm_logs/train_arm_%j.log

echo 'Training on ARM architecture started at ' $(date)
echo 'Hostname: ' $(hostname)
echo 'Running on partition: arm_fast (Fujitsu A64FX, aarch64, 32GB RAM)'
echo ''

# Set nnU-Net directories
export nnUNet_raw="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_results"
export PYTHONPATH="${HOME}/Bachelor-Thesis:${PYTHONPATH}"

# Load HPE Cray Programming Environment modules for ARM
module load cray-ccdb
module load cray-mvapich2_pmix_nogpu

echo 'Loaded modules:'
module list
echo ''

cd "${HOME}/Bachelor-Thesis"

# Train model on ARM
# Note: ARM training may be slower than x86_64
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans --c

echo 'Training finished at ' $(date)
