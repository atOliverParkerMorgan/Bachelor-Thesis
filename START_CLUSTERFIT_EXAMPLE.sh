#!/bin/bash
# ClusterFIT Pipeline - Complete Usage Example
# This script demonstrates the typical workflow for using nn_UNet on ClusterFIT

# Prerequisites:
# 1. SSH access to ClusterFIT (cluster.in.fit.cvut.cz)
# 2. Project cloned on ClusterFIT
# 3. Poetry installed and configured
# 4. SSH key added to GitLab profile
# 5. Connected to FIT VPN (if outside network)

set -e  # Exit on any error

PROJECT_DIR="$HOME/Bachelor-Thesis"
cd "$PROJECT_DIR" || { echo "Project directory not found"; exit 1; }

echo "========================================"
echo "nn_UNet ClusterFIT Pipeline Example"
echo "========================================"
echo ""

# Configuration
DATASET_ID=1
DATASET_NAME="BPWoodDefects"
CONFIGURATION="3d_fullres"
FOLD="0"

echo "Configuration:"
echo "  Dataset ID: $DATASET_ID"
echo "  Dataset Name: $DATASET_NAME"
echo "  Configuration: $CONFIGURATION"
echo "  Fold: $FOLD"
echo ""

# ============================================================
# STEP 1: Dry Run - Test the planning command
# ============================================================
echo "STEP 1: Testing plan command (dry run)"
echo "Command: poetry run python -m src.nn_UNet.pipeline plan --clusterfit --slurm-dry-run"
echo ""
echo "This shows what command will be submitted without actually submitting it."
echo "Press Enter to continue..."
read -r

poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-dry-run \
  --slurm-partition fast \
  --slurm-cpus-per-task 8

echo ""
echo "✓ Dry run successful!"
echo ""

# ============================================================
# STEP 2: Actually submit planning job
# ============================================================
echo "STEP 2: Submitting planning job (CPU, parallelized)"
echo ""

PLAN_JOB_ID=$(poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00 \
  --slurm-job-name "nnunet-plan" \
  2>&1 | grep -oP 'Job submitted successfully with ID: \K[0-9]+')

echo "✓ Planning job submitted with ID: $PLAN_JOB_ID"
echo ""
echo "Monitor with: squeue -j $PLAN_JOB_ID"
echo "View logs: tail -f slurm_logs/nnunet-plan_${PLAN_JOB_ID}.log"
echo ""
echo "Waiting for planning to complete..."

# Option to wait or continue
echo "Do you want to:"
echo "  1. Wait for planning to complete (recommended for first use)"
echo "  2. Continue without waiting"
echo ""
read -p "Enter choice (1 or 2): " CHOICE

if [ "$CHOICE" = "1" ]; then
    echo "Waiting for job $PLAN_JOB_ID to complete..."
    poetry run python -m src.nn_UNet.pipeline plan \
      --clusterfit \
      --slurm-partition fast \
      --slurm-wait
    echo "✓ Planning completed!"
else
    echo "Skipping wait. You can monitor the job with 'squeue -j $PLAN_JOB_ID'"
    echo ""
    echo "When planning is complete, run the next step."
fi

echo ""

# ============================================================
# STEP 3: Submit training job
# ============================================================
echo "STEP 3: Submitting training job (GPU)"
echo ""

TRAIN_JOB_ID=$(poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 12:00:00 \
  --slurm-cpus-per-task 4 \
  --slurm-job-name "nnunet-train" \
  --configuration "$CONFIGURATION" \
  --fold "$FOLD" \
  2>&1 | grep -oP 'Job submitted successfully with ID: \K[0-9]+')

echo "✓ Training job submitted with ID: $TRAIN_JOB_ID"
echo ""
echo "Monitor with: squeue -j $TRAIN_JOB_ID"
echo "View logs: tail -f slurm_logs/nnunet-train_${TRAIN_JOB_ID}.log"
echo ""
echo "Training typically takes 8-24 hours."
echo "You can close the SSH connection and check status later."
echo ""
echo "Later, check job status:"
echo "  $ squeue -j $TRAIN_JOB_ID"
echo ""
echo "After training completes, run inference:"
echo "  $ poetry run python -m src.nn_UNet.pipeline predict --input ./images --output ./results --clusterfit"
echo ""

# ============================================================
# Summary
# ============================================================
echo "========================================"
echo "Workflow Summary"
echo "========================================"
echo ""
echo "Submitted jobs:"
if [ -n "$PLAN_JOB_ID" ]; then
    echo "  - Planning:  Job ID $PLAN_JOB_ID"
fi
if [ -n "$TRAIN_JOB_ID" ]; then
    echo "  - Training:  Job ID $TRAIN_JOB_ID"
fi
echo ""
echo "Next steps:"
echo "  1. Monitor job status: squeue -u \$USER"
echo "  2. Check specific job: squeue -j <JOB_ID>"
echo "  3. View logs: tail -f slurm_logs/<logfile>"
echo "  4. After training, submit prediction job"
echo ""
echo "Useful commands:"
echo "  sinfo              # Check queue status"
echo "  squeue -u \$USER   # List your jobs"
echo "  scancel <JOB_ID>   # Cancel job"
echo ""
echo "Documentation:"
echo "  - Full guide: CLUSTERFIT_GUIDE.md"
echo "  - Quick ref:  CLUSTERFIT_QUICK_REF.md"
echo ""
echo "✓ Script complete!"
