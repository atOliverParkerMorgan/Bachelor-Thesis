#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:p100_16:1
#SBATCH --time=12:00:00
#SBATCH --job-name=nnunet-train-p100
#SBATCH --output=slurm_logs/train_p100_%j.log

echo 'Training started at ' $(date)
echo 'Hostname: ' $(hostname)
echo 'Running on partition: gpu (NVIDIA Tesla P100 16GB)'
echo ''

# Set nnU-Net directories
export nnUNet_raw="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="${HOME}/Bachelor-Thesis/datasets/nnunet_data/nnUNet_results"
export PYTHONPATH="${HOME}/Bachelor-Thesis:${PYTHONPATH}"

echo 'GPU info:'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
echo ''

cd "${HOME}/Bachelor-Thesis"
nnUNetv2_train 1 3d_fullres 0 -p nnUNetResEncUNetLPlans --c

echo 'Training finished at ' $(date)
