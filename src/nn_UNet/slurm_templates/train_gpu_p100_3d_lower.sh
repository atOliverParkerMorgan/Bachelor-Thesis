#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100_16:1
#SBATCH --time=48:00:00
#SBATCH --job-name=nnunet-train-3d-lower-p100
#SBATCH --output=slurm_logs/train_3d_lower_p100_%j.log

echo 'Training (3D Lower) on P100 started at ' $(date)
echo 'Hostname: ' $(hostname)
echo 'Running on partition: gpu (NVIDIA Tesla P100 16GB)'
echo 'Configuration: 3d_lower - Lower resolution for memory efficiency'
echo 'Note: P100 is slower, expect 48+ hours for same task vs 24h on A100'
echo ''

# Set nnU-Net directories
export nnUNet_raw="${HOME}/Bachelor-Thesis/src/nn_UNet/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/Bachelor-Thesis/src/nn_UNet/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="${HOME}/Bachelor-Thesis/src/nn_UNet/nnunet_data/nnUNet_results"
export PYTHONPATH="${HOME}/Bachelor-Thesis:${PYTHONPATH}"

echo 'GPU info:'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ''

cd "${HOME}/Bachelor-Thesis"

# Train model in 3d_lower configuration
# 3d_lower = lower resolution than 3d_fullres (more memory efficient, faster)
# On P100 (budget GPU), batches may be smaller but training still succeeds
nnUNetv2_train 1 3d_lower 0 -p nnUNetResEncUNetLPlans

echo 'Training finished at ' $(date)
