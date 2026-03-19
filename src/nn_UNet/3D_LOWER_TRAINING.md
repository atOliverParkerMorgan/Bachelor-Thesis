# 3D Lower Resolution nnUNet Training on ClusterFIT

This guide explains how to train nnUNet models using the **3d_lower** configuration on ClusterFIT.

## Quick Answer

Run your training with `--configuration 3d_lower`:

```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower \
  --fold 0
```

---

## What is `3d_lower`?

### nnUNet Configurations Explained

nnUNet uses different preprocessing configurations for different use cases:

| Configuration | Resolution | Memory | Speed | Use Case |
|---------------|-----------|--------|-------|----------|
| **3d_fullres** | Full resolution | High | Slow | Maximum accuracy, when GPU memory allows |
| **3d_lower** | 50% of full res | Medium | Medium | Balanced accuracy/speed (default) |
| **2d** | 2D slices | Low | Fast | 2D images or very memory-constrained |

### Benefits of 3d_lower

✅ **Memory Efficient**
- Uses ~50% less VRAM than 3d_fullres
- Runs on P100 (16GB) and V100 (32GB) GPUs
- Good for complex models or large datasets

✅ **Faster Training**
- Quicker iterations due to lower resolution
- Faster inference at deployment time
- Less computational overhead

✅ **Balanced Quality**
- Still maintains high accuracy
- Most common configuration for production
- Good starting point for most projects

### Trade-offs

⚠️ **Slightly Lower Accuracy**
- Resolution reduction means some fine details are lost
- Still ~95-98% of full-res accuracy in most cases
- Usually not noticeable in clinical practice

⚠️ **Inference on Lower Res**
- Predictions are made on lower resolution
- Upsampled back to original size
- May miss very small features

---

## Training Commands

### Simple: On GPU (Any Model Available)

```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --configuration 3d_lower
```

### Recommended: A100 40GB (Best Balance)

```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 24:00:00 \
  --configuration 3d_lower \
  --fold 0
```

### Economy: P100 16GB (Memory Constrained)

```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu p100 \
  --slurm-time 48:00:00 \
  --configuration 3d_lower \
  --fold 0
```

### All Folds: 5-Fold Cross-Validation

```bash
# Fold 0
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower \
  --fold 0

# Fold 1
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower 
  --fold 1

# ... repeat for folds 2, 3, 4
```

### Resume Previous Training

```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower 
  --fold 0 \
  --continue-training
```

---

## Full Workflow for 3D Lower

### Step 1: Prepare Dataset (Local or CPU)

```bash
poetry run python -m src.nn_UNet.pipeline prepare
```

### Step 2: Planning & Preprocessing

```bash
# On CPU (parallelized) - plan both 3d_fullres and 3d_lower
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00 \
  --plan-configurations 3d_fullres 3d_lower
```

Or plan only 3d_lower:
```bash
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --plan-configurations 3d_lower
```

### Step 3: Train (GPU)

```bash
# Fold 0
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 24:00:00 \
  --configuration 3d_lower 
  --fold 0 \
  --slurm-job-name "nnunet-3d-lower-fold0"

# Fold 1
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 24:00:00 \
  --configuration 3d_lower 
  --fold 1 \
  --slurm-job-name "nnunet-3d-lower-fold1"
```

### Step 4: Inference

```bash
# After all training is complete
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images \
  --output ./predictions \
  --configuration 3d_lower \
  --clusterfit \
  --slurm-partition gpu
```

---

## GPU Selection Guide for 3D Lower

### Recommended

**Tesla A100 40GB** ⭐
```bash
--slurm-gpu a100_40 --slurm-time 24:00:00
```
- Best speed/availability balance
- 24h typical training time
- Fast queue (relatively)

### Good Alternative

**Tesla V100 32GB**
```bash
--slurm-gpu v100 --slurm-time 36:00:00
```
- Sufficient memory for 3d_lower
- Slower than A100 but older GPUs tend to queue faster
- Good for cost/time tradeoff

### Budget Option

**Tesla P100 16GB**
```bash
--slurm-gpu p100 --slurm-time 48:00:00
```
- Still works for 3d_lower (more conservative batches)
- Significant slowdown (2-3x vs A100)
- Good for quick testing/small datasets

### Not Recommended

**A100 80GB**
- Overkill for 3d_lower (wastes resources)
- Better for 3d_fullres or distributed training

---

## Monitoring Training

### Check Job Status

```bash
# SSH to ClusterFIT first
ssh cluster.in.fit.cvut.cz

# List all your jobs
squeue -u $USER

# Check specific job
squeue -j 123456

# View job details
scontrol show job 123456
```

### Monitor Training Progress

```bash
# Real-time log
tail -f slurm_logs/train_3d_lower_123456.log

# Or check nnU-Net training log
tail -f src/nn_UNet/nnunet_data/nnUNet_results/Dataset001_BPWoodDefects/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_lower/fold_0/training_log_0.txt
```

### Check GPU Usage

```bash
# While job is running on compute node
nvidia-smi
```

### Training Time Estimates (A100 40GB)

| Dataset Size | Epochs | Time Estimate |
|---|---|---|
| Small (<1000 images) | 1000 | 6-8 hours |
| Medium (1000-5000 images) | 1000 | 12-18 hours |
| Large (5000+ images) | 1000 | 24-48 hours |

---

## Common Issues & Solutions

### Issue: "CUDA out of memory"

Even on A100, with faulty configs this can happen.

**Solution**:
```bash
# Use planning-determined batch size
# Should be automatic, but if issues:

# Reduce CPUs to reduce parallelism
--slurm-cpus-per-task 2

# Or switch to larger GPU
--slurm-gpu a100_80

# Or use 2d instead
--configuration 2d
```

### Issue: "FileNotFoundError: plans_identifier not found"

Happens if planning didn't complete successfully.

**Solution**:
```bash
# Make sure planning finished
squeue -j <PLAN_JOB_ID>

# Rerun planning
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-wait \
  --plan-configurations 3d_lower

# Then retry training
```

### Issue: Job in queue too long

Happens with limited GPU availability.

**Solution**:
```bash
# Try P100 (queues faster)
--slurm-gpu p100

# Or use faster queue times
--slurm-time 36:00:00  # Instead of 48:00:00

# Submit during off-peak hours
```

### Issue: "Configuration 3d_lower not available"

Plans for 3d_lower weren't created during planning step.

**Solution**:
```bash
# Ensure 3d_lower was planned
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --plan-configurations 3d_fullres 3d_lower \
  --slurm-wait

# Then train
```

---

## 3D Lower vs Others: Quick Comparison

### Use 3d_lower When:
- ✅ You have 16-40GB GPU
- ✅ You want balanced accuracy/speed
- ✅ You're training for clinical use (not research)
- ✅ You need deployment on modest hardware
- ✅ You don't need every fine detail

### Use 3d_fullres When:
- ✅ You have unlimited A100 80GB or multiple GPUs
- ✅ You need maximum accuracy (research)
- ✅ Your dataset is very small (<100 images)
- ✅ Fine details are critical (e.g., tiny lesions)

### Use 2d When:
- ✅ Your data is truly 2D
- ✅ Memory is severely limited (P100 with large batch)
- ✅ Quick prototyping needed
- ✅ Training time is critical (testing)

---

## Advanced: Multiple Folds Batch Submission

Submit all 5 folds at once:

```bash
#!/bin/bash
# submit_all_folds.sh

for fold in 0 1 2 3 4; do
  echo "Submitting fold $fold..."
  poetry run python -m src.nn_UNet.pipeline train \
    --clusterfit \
    --slurm-partition gpu \
    --slurm-gpu a100_40 \
    --slurm-time 24:00:00 \
    --configuration 3d_lower \
    --fold "$fold" \
    --slurm-job-name "nnunet-3d-lower-fold$fold"
  
  # Small delay to avoid overwhelming scheduler
  sleep 2
done

echo "All folds submitted!"
```

Run with:
```bash
chmod +x submit_all_folds.sh
./submit_all_folds.sh
```

---

## Ready-Made Templates

Pre-built scripts in `src/nn_UNet/slurm_templates/`:

| Script | Configuration | GPU |
|--------|---|---|
| `train_gpu_a100_3d_lower.sh` | 3d_lower | A100 40GB |
| `train_gpu_a100.sh` | 3d_fullres | A100 40GB |
| `train_gpu_p100.sh` | 3d_fullres | P100 16GB |

Use as reference or copy & modify for your needs.

---

## Summary

### Simplest Command
```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --configuration 3d_lower
```

### Best Practices Command
```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 24:00:00 \
  --configuration 3d_lower \
  --fold 0 \
  --slurm-job-name "3d-lower-fold0" \
  --slurm-email your@email.com
```

### Check Status
```bash
# On ClusterFIT
squeue -u $USER
tail -f slurm_logs/train_3d_lower_*.log
```

---

**Next Steps**:
1. SSH to ClusterFIT
2. Run planning if not done: `poetry run python -m src.nn_UNet.pipeline plan --clusterfit --plan-configurations 3d_lower --slurm-wait`
3. Submit training: `poetry run python -m src.nn_UNet.pipeline train --clusterfit --configuration 3d_lower`
4. Monitor with `squeue -u $USER`

---

**Documentation**: See [CLUSTERFIT_GUIDE.md](CLUSTERFIT_GUIDE.md) for general ClusterFIT usage
