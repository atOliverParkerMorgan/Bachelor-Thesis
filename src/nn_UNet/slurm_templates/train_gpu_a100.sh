#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100_40:1
#SBATCH --time=12:00:00
#SBATCH --job-name=nnunet-train
#SBATCH --output=slurm_logs/train_%j.log

echo 'Training started at ' $(date)
echo 'Hostname: ' $(hostname)
echo 'Running on partition: gpu (NVIDIA Tesla A100 40GB)'
echo ''

# Set nnU-Net directories
export nnUNet_raw="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_results"
export PYTHONPATH="${HOME}/Bachelor-Thesis:${PYTHONPATH}"

# Optional: Set training parameters
# export NNUNET_SAVE_EVERY=10
# export NNUNET_INITIAL_LR=1e-2
# export NNUNET_SKIP_ARCH_PLOT=1

echo 'GPU info:'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ''

cd "${HOME}/Bachelor-Thesis"

# Train model: dataset_id, configuration, fold, and plans_identifier
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans --c

echo 'Training finished at ' $(date)
