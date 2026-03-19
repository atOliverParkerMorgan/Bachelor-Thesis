# 3D Lower Training - Quick Cheat Sheet

**TL;DR**: Add `--configuration 3d_lower` to your training command.

## One-Line Commands

### Plan (CPU, ~30 min - 1 hour)
```bash
poetry run python -m src.nn_UNet.pipeline plan --clusterfit --plan-configurations 3d_lower --slurm-wait
```

### Train (GPU, A100 recommended)
```bash
poetry run python -m src.nn_UNet.pipeline train --clusterfit --slurm-gpu a100_40 --configuration 3d_lower --fold 0
```

### Predict
```bash
poetry run python -m src.nn_UNet.pipeline predict --input ./images --output ./results --configuration 3d_lower --clusterfit
```

---

## GPU Options for 3D Lower

| GPU | Command | Time | Cost |
|-----|---------|------|------|
| **A100 40GB** ⭐ | `--slurm-gpu a100_40` | 24h | Medium |
| V100 32GB | `--slurm-gpu v100` | 36h | Medium |
| P100 16GB | `--slurm-gpu p100` | 48h | Low |

---

## Configuration Comparison

| Config | Resolution | Memory | Speed | Accuracy |
|---|---|---|---|---|
| **3d_lower** | 50% | Medium ✓ | Medium ✓ | Very good ✓ |
| 3d_fullres | 100% | High | Slow | Best |
| 2d | 2D slices | Low | Fast | Good for 2D |

---

## Full Commands

### Step 1: Plan for 3d_lower
```bash
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --plan-configurations 3d_lower
```

### Step 2: Train Fold 0
```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --configuration 3d_lower \
  --fold 0
```

### Step 3: Train All Folds (5-fold CV)
```bash
for fold in 0 1 2 3 4; do
  poetry run python -m src.nn_UNet.pipeline train \
    --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
    --configuration 3d_lower --fold "$fold"
done
```

### Step 4: Predict
```bash
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test --output ./results \
  --configuration 3d_lower --clusterfit
```

---

## Monitoring

```bash
# List jobs
squeue -u $USER

# Specific job
squeue -j 123456

# Cancel
scancel 123456

# Logs
tail -f slurm_logs/train_3d_lower_*.log
```

---

## Templates

Pre-built scripts in `src/nn_UNet/slurm_templates/`:
- `train_gpu_a100_3d_lower.sh` ← Use this!
- `train_gpu_p100_3d_lower.sh`

Copy & modify:
```bash
cp src/nn_UNet/slurm_templates/train_gpu_a100_3d_lower.sh my_train.sh
nano my_train.sh
sbatch my_train.sh
```

---

## Common Issues

| Issue | Solution |
|---|---|
| Out of memory | Use `--slurm-gpu a100_80` or plan 2d |
| Can't find 3d_lower | Rerun planning: `pipeline plan --plan-configurations 3d_lower` |
| Job stuck in queue | Use `--slurm-gpu p100` (queues faster) |
| Training too slow | Use `--slurm-gpu a100_40` (not P100) |

---

## Time Estimates (A100 40GB, 3d_lower)

- **Small dataset** (<1000 img): 6-8h
- **Medium dataset** (1000-5000 img): 12-18h  
- **Large dataset** (5000+ img): 24-36h
- **5-fold training**: Multiply above by 5

---

## Files & Docs

| Document | What It Has |
|---|---|
| **3D_LOWER_TRAINING.md** ← START HERE | Complete guide |
| 3D_LOWER_COMMANDS.sh | Copy-paste commands |
| CLUSTERFIT_GUIDE.md | General ClusterFIT help |
| slurm_templates/\*.sh | Example scripts |

---

## Full Workflow Example

```bash
# 1. SSH to ClusterFIT
ssh cluster.in.fit.cvut.cz

# 2. Navigate to project
cd ~/Bachelor-Thesis

# 3. Check planning done
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit --plan-configurations 3d_lower --slurm-wait

# 4. Start training (all 5 folds)
for fold in 0 1 2 3 4; do
  poetry run python -m src.nn_UNet.pipeline train \
    --clusterfit --slurm-gpu a100_40 \
    --configuration 3d_lower --fold "$fold" \
    --slurm-job-name "3d-lower-fold$fold"
done

# 5. Monitor
watch 'squeue -u $USER'

# 6. After training completes, predict
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images --output ./results \
  --configuration 3d_lower --clusterfit

# 7. Download results
scp -r cluster.in.fit.cvut.cz:~/Bachelor-Thesis/results .
```

---

## Key Differences: 3d_lower vs 3d_fullres

**3d_lower** (Recommended for most cases):
- ✅ 50% lower memory requirement
- ✅ Faster training (4-6h vs 8-12h on A100)
- ✅ Runs on P100/V100 without OOM
- ✅ Still ~95-98% accuracy of fullres
- ⚠️ Lower resolution = may miss tiny features

**3d_fullres** (Maximum accuracy):
- ✅ Maximum accuracy
- ✅ Capture all details
- ⚠️ Requires A100 40GB+ or 80GB
- ⚠️ Takes 2x longer to train
- ⚠️ May fail on P100/V100

---

## Parameters Summary

```bash
--configuration 3d_lower   # ALWAYS USE THIS for 3d_lower
--fold 0-4                 # Which fold (default: 0)
--slurm-gpu a100_40       # GPU model (p100, v100, a100_40, a100_80)
--slurm-time 24:00:00     # Time limit (default: 12h, use 24h for safety)
--slurm-partition gpu     # Always "gpu" for training
--slurm-job-name NAME     # Human-readable name
--slurm-email EMAIL       # Notify when done
--slurm-wait              # Don't return until done
--continue-training       # Resume from checkpoint
```

---

**Need help?** → Read `3D_LOWER_TRAINING.md` (full guide with examples)
