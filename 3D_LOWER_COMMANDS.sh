#!/usr/bin/env bash
# 3D Lower nnUNet Training - Copy & Paste Commands
# All commands ready to run on ClusterFIT

# ============================================================
# PREPARE PHASE (Run Locally or Submit to CPU)
# ============================================================

# Plan for 3d_lower only
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00 \
  --plan-configurations 3d_lower \
  --slurm-job-name "nnunet-plan"

# Plan for both 3d_fullres and 3d_lower
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00 \
  --plan-configurations 3d_fullres 3d_lower \
  --slurm-wait

# ============================================================
# TRAIN PHASE (GPU)
# ============================================================

# --- SIMPLE: Any available GPU
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --configuration 3d_lower \
  --fold 0

# --- RECOMMENDED: A100 40GB (best balance)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 24:00:00 \
  --configuration 3d_lower \
  --fold 0 \
  --slurm-job-name "3d-lower-fold0"

# --- BUDGET: P100 (slower, queues faster)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu p100 \
  --slurm-time 48:00:00 \
  --configuration 3d_lower \
  --fold 0

# --- WITH EMAIL: Get notified when done
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower \
  --fold 0 \
  --slurm-email your.email@fit.cvut.cz

# --- CONTINUE FROM CHECKPOINT
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower 
  --fold 0 \
  --continue-training

# ============================================================
# MULTI-FOLD TRAINING (5-Fold Cross-Validation)
# ============================================================

# Fold 0
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
  --configuration 3d_lower --fold 0

# Fold 1
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
  --configuration 3d_lower --fold 1

# Fold 2
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
  --configuration 3d_lower --fold 2

# Fold 3
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
  --configuration 3d_lower --fold 3

# Fold 4
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
  --configuration 3d_lower --fold 4

# ============================================================
# INFERENCE PHASE (GPU)
# ============================================================

# Simple inference
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images \
  --output ./predictions \
  --configuration 3d_lower \
  --clusterfit

# With specific GPU
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images \
  --output ./predictions \
  --configuration 3d_lower \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 02:00:00

# ============================================================
# MONITORING (SSH to ClusterFIT first)
# ============================================================

# List all your jobs
squeue -u $USER

# Check specific job
squeue -j 123456

# Job full info
scontrol show job 123456

# Cancel job
scancel 123456

# View training logs
tail -f slurm_logs/train_3d_lower_*.log

# Check GPU usage during job
ssh cf-prod-node-027  # (replace with your assigned node)
nvidia-smi

# ============================================================
# TESTING: DRY RUN (no submission)
# ============================================================

# See the command without submitting
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-dry-run \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower

# ============================================================
# USEFUL: WAIT FOR COMPLETION
# ============================================================

# Command that blocks until job finishes
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-wait \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower 
  --fold 0

# Then immediately run inference
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images \
  --output ./predictions \
  --configuration 3d_lower \
  --clusterfit

# ============================================================
# BATCH SCRIPT: Auto-submit all 5 folds
# ============================================================

#!/bin/bash
# save as: submit_3d_lower_all_folds.sh

for fold in 0 1 2 3 4; do
  echo "Submitting fold $fold..."
  poetry run python -m src.nn_UNet.pipeline train \
    --clusterfit \
    --slurm-partition gpu \
    --slurm-gpu a100_40 \
    --slurm-time 24:00:00 \
    --configuration 3d_lower \
    --fold "$fold" \
    --slurm-job-name "3d-lower-fold$fold"
  sleep 2
done

# Run with:
# chmod +x submit_3d_lower_all_folds.sh
# ./submit_3d_lower_all_folds.sh

# ============================================================
# FULL PIPELINE: Prepare + Plan + Train + Predict
# ============================================================

# 1. Prepare dataset
poetry run python -m src.nn_UNet.pipeline prepare

# 2. Plan (wait for completion)
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-plan-configurations 3d_lower \
  --slurm-wait

# 3. Train fold 0 (wait for completion)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower 
  --fold 0 \
  --slurm-wait

# 4. Predict
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images \
  --output ./predictions \
  --configuration 3d_lower \
  --clusterfit

# ============================================================
# TEMPLATES: Use pre-built scripts
# ============================================================

# Copy template and edit
cp src/nn_UNet/slurm_templates/train_gpu_a100_3d_lower.sh my_train_3d_lower.sh
nano my_train_3d_lower.sh  # Edit as needed
sbatch my_train_3d_lower.sh

# Available templates:
# - src/nn_UNet/slurm_templates/train_gpu_a100_3d_lower.sh
# - src/nn_UNet/slurm_templates/train_gpu_p100_3d_lower.sh

# ============================================================
# QUICK REFERENCE: GPU TIME ESTIMATES (A100 40GB)
# ============================================================

# Small dataset (<500 images):
# - Planning: 15-30 min
# - Training (1 fold): 6-8 hours
# - Total: ~8 hours

# Medium dataset (500-2000 images):
# - Planning: 30 min - 1 hour
# - Training (1 fold): 12-18 hours
# - Total: ~13-19 hours

# Large dataset (2000+ images):
# - Planning: 1-2 hours
# - Training (1 fold): 24-36 hours
# - Total: ~25-38 hours

# 5-Fold training:
# - Run sequentially: ~24h (fold 0) + ~24h (fold 1) + ... = multiple days
# - Run parallel: All at once if quota allows, watch job queue

# ============================================================
# COMMON PARAMETERS EXPLAINED
# ============================================================

# --configuration 3d_lower    | Use 3D lower resolution config
# --fold 0                    | Use fold 0 (0-4 for 5-fold CV)
# --slurm-partition gpu       | Use GPU partition
# --slurm-gpu a100_40        | Specific GPU model
# --slurm-time 24:00:00      | Time limit (HH:MM:SS)
# --slurm-job-name NAME      | Human-readable job name
# --slurm-email EMAIL        | Email when done
# --slurm-wait               | Wait for completion
# --slurm-dry-run            | Test without submitting
# --continue-training        | Resume from checkpoint

# ============================================================
# NEED HELP?
# ============================================================

# Full guide:
# cat src/nn_UNet/3D_LOWER_TRAINING.md

# General ClusterFIT:
# cat src/nn_UNet/CLUSTERFIT_GUIDE.md

# Quick reference:
# cat src/nn_UNet/CLUSTERFIT_QUICK_REF.md
