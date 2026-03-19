#!/usr/bin/env bash
# ============================================================
# 3D LOWER nnUNet TRAINING - COMPLETE SETUP GUIDE
# ============================================================
# This file shows the exact workflow to train 3d_lower on ClusterFIT
# Copy commands directly to your terminal on ClusterFIT
# ============================================================

echo "
╔════════════════════════════════════════════════════════════╗
║  3D LOWER nnUNet Training on ClusterFIT                    ║
║  Complete Workflow                                          ║
╚════════════════════════════════════════════════════════════╝
"

# ============================================================
# REQUIREMENTS
# ============================================================
echo "
REQUIREMENTS:
✓ SSH access to ClusterFIT
✓ Project cloned at ~/Bachelor-Thesis
✓ Poetry environment set up
✓ Connected to FIT VPN (if outside)
✓ SSH key added to GitLab profile
"

# ============================================================
# STEP 1: CONNECT
# ============================================================
echo "
STEP 1: Connect to ClusterFIT
────────────────────────────────────────────────────────────"
echo "
Command:
  ssh -i ~/.ssh/id_clusterfit username@cluster.in.fit.cvut.cz

Or if using default SSH key:
  ssh username@cluster.in.fit.cvut.cz
"

# ============================================================
# STEP 2: NAVIGATE TO PROJECT
# ============================================================
echo "
STEP 2: Navigate to Project
────────────────────────────────────────────────────────────"
echo "
Command:
  cd ~/Bachelor-Thesis
"

# ============================================================
# STEP 3: CHECK PLANNING (PREPROCESSING)
# ============================================================
echo "
STEP 3: Make Sure Planning is Done for 3d_lower
────────────────────────────────────────────────────────────"
echo "
The 3d_lower configuration needs to be preprocessed first.
Run this once (it takes 30 min - 1 hour):
"
echo "Command:"
cat << 'EOF'
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00 \
  --plan-configurations 3d_lower \
  --slurm-wait
EOF
echo ""
echo "This will:"
echo "  ✓ Submit planning job to CPU partition"
echo "  ✓ Wait until it finishes"
echo "  ✓ Download and process dataset"
echo "  ✓ Create 3d_lower configuration files"
echo ""

# ============================================================
# STEP 4: TRAIN - OPTION A (SIMPLE)
# ============================================================
echo "
STEP 4: Train 3D Lower Model
────────────────────────────────────────────────────────────"
echo ""
echo "OPTION A: Simplest (use any available GPU)"
echo "───────────────────────────────────────────"
echo "
Command:
"
cat << 'EOF'
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --configuration 3d_lower \
  --fold 0
EOF
echo ""
echo "Results:"
echo "  ✓ Submits to GPU queue"
echo "  ✓ Trains on 3d_lower configuration"
echo "  ✓ Fold 0 (first fold)"
echo "  ✓ ~12 hour default time"
echo ""

# ============================================================
# STEP 4B: TRAIN - OPTION B (RECOMMENDED)
# ============================================================
echo ""
echo "OPTION B: With Specific GPU (RECOMMENDED)"
echo "──────────────────────────────────────────"
echo "
Command:
"
cat << 'EOF'
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 24:00:00 \
  --configuration 3d_lower \
  --fold 0 \
  --slurm-job-name "3d-lower-fold0"
EOF
echo ""
echo "Results with A100 40GB:"
echo "  ✓ Submits to GPU partition"
echo "  ✓ Gets Tesla A100 40GB GPU"
echo "  ✓ 24 hour time limit (safe for medium datasets)"
echo "  ✓ Job name: 3d-lower-fold0 (easy to track)"
echo "  ✓ Training time: ~24 hours"
echo ""

# ============================================================
# STEP 4C: TRAIN - OPTION C (BUDGET)
# ============================================================
echo ""
echo "OPTION C: Budget GPU (Slower, Queues Faster)"
echo "───────────────────────────────────────────"
echo "
Command:
"
cat << 'EOF'
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu p100 \
  --slurm-time 48:00:00 \
  --configuration 3d_lower \
  --fold 0
EOF
echo ""
echo "Results with P100 16GB:"
echo "  ✓ Cheaper GPU, queues faster"
echo "  ✓ Training time: ~48 hours (2x slower)"
echo "  ✓ Good for cost-sensitive runs"
echo ""

# ============================================================
# STEP 5: MONITOR TRAINING
# ============================================================
echo ""
echo "STEP 5: Monitor Training Job
────────────────────────────────────────────────────────────"
echo ""
echo "Check all your jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job:"
echo "  squeue -j 123456"
echo ""
echo "Watch in real-time:"
echo "  watch 'squeue -u \$USER'"
echo ""
echo "View training logs:"
echo "  tail -f slurm_logs/train_3d_lower_*.log"
echo ""
echo "Cancel if needed:"
echo "  scancel 123456"
echo ""

# ============================================================
# STEP 6: MULTI-FOLD TRAINING (5-FOLD CV)
# ============================================================
echo ""
echo "STEP 6: Train All 5 Folds (Optional)
────────────────────────────────────────────────────────────"
echo "
For cross-validation, train all 5 folds:
"
cat << 'EOF'
for fold in 0 1 2 3 4; do
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
EOF
echo ""
echo "This will:"
echo "  ✓ Submit 5 training jobs (one per fold)"
echo "  ✓ All run in parallel in the queue"
echo "  ✓ Take ~24h each (can overlap)"
echo "  ✓ Total time: ~1-2 days for all 5 folds"
echo ""

# ============================================================
# STEP 7: PREDICT (AFTER TRAINING)
# ============================================================
echo ""
echo "STEP 7: Run Predictions (After Training Completes)
────────────────────────────────────────────────────────────"
echo "
Once training is done, run inference:
"
cat << 'EOF'
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images \
  --output ./predictions \
  --configuration 3d_lower \
  --clusterfit \
  --slurm-partition gpu
EOF
echo ""
echo "This will:"
echo "  ✓ Use trained model"
echo "  ✓ Predict on images in ./test_images"
echo "  ✓ Save predictions to ./predictions"
echo ""

# ============================================================
# QUICK REFERENCE TABLE
# ============================================================
echo ""
echo "QUICK REFERENCE: 3D LOWER CONFIGURATION
────────────────────────────────────────────────────────────"
echo ""
cat << 'EOF'
╭─────────────────┬──────────────┬──────────────┬────────────────╮
│ GPU             │ VRAM         │ Time (3dL)   │ Cost           │
├─────────────────┼──────────────┼──────────────┼────────────────┤
│ A100 40GB  ⭐   │ 40GB         │ ~24h/fold    │ Medium         │
│ V100 32GB       │ 32GB         │ ~36h/fold    │ Medium         │
│ P100 16GB       │ 16GB         │ ~48h/fold    │ Low            │
│ A100 80GB       │ 80GB         │ ~20h/fold    │ High (watch!)  │
╰─────────────────┴──────────────┴──────────────┴────────────────╯
EOF
echo ""

# ============================================================
# TROUBLESHOOTING
# ============================================================
echo ""
echo "TROUBLESHOOTING
────────────────────────────────────────────────────────────"
echo ""
echo "Problem: 'Can't find 3d_lower configuration'"
echo "Solution: Rerun planning - it creates 3d_lower files"
echo "  poetry run python -m src.nn_UNet.pipeline plan \\"
echo "    --clusterfit --plan-configurations 3d_lower"
echo ""
echo "Problem: 'CUDA out of memory'"
echo "Solution: Use A100 80GB or reduce batch size"
echo "  --slurm-gpu a100_80"
echo ""
echo "Problem: 'Job stuck in queue'"
echo "Solution: Use P100 (queues faster) or submit off-peak"
echo "  --slurm-gpu p100"
echo ""
echo "Problem: 'Training too slow'"
echo "Solution: Use A100 40GB (not P100)"
echo "  --slurm-gpu a100_40"
echo ""

# ============================================================
# FULL WORKFLOW EXAMPLE
# ============================================================
echo ""
echo "COMPLETE WORKFLOW (Copy & Paste)
────────────────────────────────────────────────────────────"
echo ""
cat << 'EOF'
# 1. SSH and navigate
ssh cluster.in.fit.cvut.cz
cd ~/Bachelor-Thesis

# 2. Plan for 3d_lower (run once, 30-60 min)
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --plan-configurations 3d_lower \
  --slurm-wait

# 3. Train fold 0 (24h on A100)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-gpu a100_40 \
  --configuration 3d_lower \
  --fold 0

# 4. Monitor
squeue -u $USER
tail -f slurm_logs/train_3d_lower_*.log

# 5. Predict (after training)
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images \
  --output ./predictions \
  --configuration 3d_lower \
  --clusterfit
EOF
echo ""

# ============================================================
# DOCUMENTATION
# ============================================================
echo ""
echo "DOCUMENTATION FILES
────────────────────────────────────────────────────────────"
echo ""
echo "Quick Start (5 min read):"
echo "  cat 3D_LOWER_QUICK_START.md"
echo ""
echo "All Commands (copy-paste):"
echo "  cat 3D_LOWER_COMMANDS.sh"
echo ""
echo "Full Guide (detailed):"
echo "  cat src/nn_UNet/3D_LOWER_TRAINING.md"
echo ""
echo "ClusterFIT General Help:"
echo "  cat src/nn_UNet/CLUSTERFIT_GUIDE.md"
echo ""

# ============================================================
# SUCCESS INDICATORS
# ============================================================
echo ""
echo "HOW TO KNOW IT WORKED
────────────────────────────────────────────────────────────"
echo ""
echo "✓ Job ID returned: 'Job submitted successfully with ID: 12345'"
echo "✓ squeue shows job: 'squeue -u \$USER' lists it"
echo "✓ Log file created: 'ls slurm_logs/train_3d_lower_*.log'"
echo "✓ Training started: 'tail -f slurm_logs/train_3d_lower_*.log'"
echo ""

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY: Train 3D Lower on ClusterFIT                     ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║                                                            ║"
echo "║  1. SSH to ClusterFIT                                     ║"
echo "║  2. Run planning (once): --plan-configurations 3d_lower   ║"
echo "║  3. Train: --configuration 3d_lower                       ║"
echo "║  4. Monitor: squeue -u \$USER                              ║"
echo "║  5. Predict: --configuration 3d_lower                     ║"
echo "║                                                            ║"
echo "║  RECOMMENDED GPU: A100 40GB                               ║"
echo "║  TRAINING TIME: ~24 hours/fold                            ║"
echo "║  BEST FOR: Most projects (balanced quality/speed)         ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Ready to train? Run:"
echo "  ssh cluster.in.fit.cvut.cz"
echo "Then copy commands above!"
echo ""
