#!/bin/bash
#SBATCH --partition=fast
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=nnunet-plan
#SBATCH --output=slurm_logs/plan_%j.log

echo 'Planning started at ' $(date)
echo 'Hostname: ' $(hostname)
echo 'Running on partition: fast (CPU, amd64/x86_64, 64GB RAM)'
echo ''

# Set nnU-Net directories (adjust paths as needed)
export nnUNet_raw="${HOME}/Bachelor-Thesis/src/nn_UNet/nnunet_data/nnUNet_raw"
export nnUNet_preprocessed="${HOME}/Bachelor-Thesis/src/nn_UNet/nnunet_data/nnUNet_preprocessed"
export nnUNet_results="${HOME}/Bachelor-Thesis/src/nn_UNet/nnunet_data/nnUNet_results"
export PYTHONPATH="${HOME}/Bachelor-Thesis:${PYTHONPATH}"

mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

# Run planning with parallelization
# Adjust -np (number of processes) based on available cores
cd "${HOME}/Bachelor-Thesis"
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL -c 3d_fullres -np 8

echo 'Planning finished at ' $(date)
