#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=nnunet-predict
#SBATCH --output=slurm_logs/predict_%j.log

echo 'Prediction started at ' $(date)
echo 'Hostname: ' $(hostname)
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

# Run prediction
# Adjust input/output paths as needed
nnUNetv2_predict \
  -i "./input_images" \
  -o "./predictions" \
  -d 1 \
  -c 3d_fullres \
  -f 0 \
  -p nnUNetResEncUNetLPlans \
  --disable_tta \
  -npp 2 -nps 2

echo 'Prediction finished at ' $(date)
