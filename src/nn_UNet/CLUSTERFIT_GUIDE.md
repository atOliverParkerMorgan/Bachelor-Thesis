# ClusterFIT (nn_UNet Pipeline) - Usage Guide

This guide explains how to use the nn_UNet pipeline on ClusterFIT, FIT ČVUT's high-performance computing platform using Slurm job scheduler.

## Table of Contents

- [System Architecture](#system-architecture)
- [Authentication](#authentication)
- [Running Jobs Locally](#running-jobs-locally)
- [Submitting Jobs to ClusterFIT](#submitting-jobs-to-clusterfit)
- [GPU Selection Guide](#gpu-selection-guide)
- [CPU Architecture Selection](#cpu-architecture-selection)
- [Examples](#examples)
- [Monitoring Jobs](#monitoring-jobs)
- [Common Issues](#common-issues)

---

## System Architecture

### Available Resources

**Frontend Nodes** (for compilation and submission):
- `cluster.in.fit.cvut.cz` - x86_64 (Intel Xeon) - **Primary endpoint**
- `cf-frontend01-prod.cls.in.fit.cvut.cz` - x86_64 (Intel Xeon)
- `cf-frontend02-prod.cls.in.fit.cvut.cz` - x86_64 (Intel Xeon)
- `cf-frontend03-prod.cls.in.fit.cvut.cz` - aarch64 (Fujitsu A64FX) - For ARM compilation
- `cf-frontend04-prod.cls.in.fit.cvut.cz` - x86_64 (Intel Xeon) - Backup for ARM jobs

**Compute Partitions**:

| Partition | Architecture | CPU | RAM | GPU | Count | Best For |
|-----------|---|---|---|---|---|---|
| `fast` | x86_64 | Intel Xeon Gold 6254 | 64GB | None | 28 nodes | CPU workloads |
| `gpu` | x86_64 | Intel Xeon | 64GB | NVIDIA (V100, A100, P100) | Multiple | GPU training/inference |
| `arm_fast` | aarch64 | Fujitsu A64FX | 32GB | None | 8 nodes | ARM-specific workloads |
| `amd` | x86_64 | AMD EPYC | 512GB | AMD Instinct MI210 (4x) | 2 nodes | Large GPU jobs |

---

## Authentication

### SSH Setup

1. **Generate SSH key** (if you don't have one):
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_clusterfit
```

2. **Add to GitLab profile**:
   - Go to FIT GitLab → Profile Settings → SSH Keys
   - Add your public key: `~/.ssh/id_clusterfit.pub`

3. **Test connection**:
```bash
ssh -i ~/.ssh/id_clusterfit username@cluster.in.fit.cvut.cz
```

### VPN Requirement

- You must be connected to FIT VPN when accessing ClusterFIT from outside the FIT network
- Setup: [FIT VPN Documentation](https://fit.cvut.cz/)

---

## Running Jobs Locally

First, ensure the pipeline works on your local machine:

```bash
# Navigate to project directory
cd ~/Bachelor-Thesis

# Plan (preprocessing)
poetry run python -m src.nn_UNet.pipeline plan

# Train
poetry run python -m src.nn_UNet.pipeline train

# Predict
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./input_images \
  --output ./predictions
```

---

## Submitting Jobs to ClusterFIT

### Basic Submission

Add `--clusterfit` flag to submit instead of running locally:

```bash
poetry run python -m src.nn_UNet.pipeline plan --clusterfit
```

### Configuration Options

Use these flags to customize Slurm job:

| Flag | Default | Examples | Description |
|------|---------|----------|---|
| `--slurm-partition` | `fast` | `fast, gpu, arm_fast, amd` | Compute partition |
| `--slurm-time` | Command-dependent | `02:00:00, 12:00:00` | Time limit (HH:MM:SS) |
| `--slurm-gpu` | None | `p100, v100, a100_40, a100_80, mi210` | Specific GPU model |
| `--slurm-mem` | Partition default | `64G, 256G, 512G` | Override memory limit |
| `--slurm-nodes` | `1` | `1, 2, 4` | Number of compute nodes |
| `--slurm-cpus-per-task` | `1` | `1, 4, 8, 16` | CPU cores per task |
| `--slurm-job-name` | Auto | `my-train-job` | Human-readable job name |
| `--slurm-output` | Auto | `./logs/job.log` | Log file path |
| `--slurm-email` | None | `user@fit.cvut.cz` | Email for notifications |
| `--slurm-dry-run` | False | N/A | Test without submitting |
| `--slurm-wait` | False | N/A | Wait for job completion |
| `--arm-hpe-cpe` | False | N/A | Load HPE modules for ARM |

---

## GPU Selection Guide

### Choose GPU Based on Task

**For nnU-Net Training:**

- **NVIDIA Tesla P100 (16GB)** - Small models or reduced batch size
  ```bash
  --slurm-partition gpu --slurm-gpu p100
  ```

- **NVIDIA Tesla V100 (32GB)** - Medium models, recommended baseline
  ```bash
  --slurm-partition gpu --slurm-gpu v100
  ```

- **NVIDIA Tesla A100 (40GB)** - Large models, faster training ✓ **RECOMMENDED**
  ```bash
  --slurm-partition gpu --slurm-gpu a100_40
  ```

- **NVIDIA Tesla A100 (80GB)** - Very large models
  ```bash
  --slurm-partition gpu --slurm-gpu a100_80
  ```

- **AMD Instinct MI210** - Experimental (4 GPUs per node)
  ```bash
  --slurm-partition amd
  ```

### GPU Quick Reference

| GPU | VRAM | Best For | Time Limit |
|---|---|---|---|
| P100 | 16GB | nnU-Net inference, small models | 12h |
| V100 | 32GB | nnU-Net training | 12h |
| A100 40GB | 40GB | nnU-Net training | 12h |
| A100 80GB | 80GB | Multi-model training | 12h |
| MI210 | ~32GB/GPU (4x) | Distributed computing | 12h |

---

## CPU Architecture Selection

### x86_64 (Intel Xeon) - Default

```bash
# Planning (CPU-intensive preprocessing)
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-time 04:00:00 \
  --slurm-cpus-per-task 8

# CPU-only training (slow, not recommended)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition fast \
  --slurm-time 24:00:00
```

### aarch64 (Fujitsu A64FX) - ARM

**Note**: ARM architecture is slower than x86_64. Use only for special cases.

```bash
# Compile for ARM
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition arm_fast \
  --slurm-time 24:00:00 \
  --arm-hpe-cpe
```

When using `--arm-hpe-cpe`, the pipeline will:
- Load HPE Cray Programming Environment modules
- Use `cray-mvapich2_pmix_nogpu` for MPI support
- Compile on `cf-frontend03-prod` (A64FX)

---

## Examples

### Example 1: Plan on CPU (Multi-threaded)

```bash
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00 \
  --slurm-job-name "plan-large-dataset"
```

Expected log output:
```
[HH:MM:SS] Submitting to ClusterFIT...
[HH:MM:SS] Slurm script written to: /tmp/tmpXXXXXX.sh
[HH:MM:SS] Partition: fast, Time: 04:00:00
[HH:MM:SS] Job submitted successfully with ID: 123456
```

### Example 2: Train on GPU (A100)

```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 12:00:00 \
  --slurm-job-name "train-3d-fullres-fold0" \
  --slurm-email your.email@fit.cvut.cz
```

### Example 3: Quick Inference on GPU

```bash
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./my_images \
  --output ./predictions \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu v100 \
  --slurm-time 02:00:00
```

### Example 4: Test Submission (Dry Run)

```bash
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-dry-run \
  --slurm-partition fast
```

Output shows script without submitting:
```
$ sbatch --test-only /tmp/tmpXXXX.sh
```

### Example 5: Large Dataset Planning

```bash
# Use a high-memory node from the 'amd' partition
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition amd \
  --slurm-nodes 1 \
  --slurm-cpus-per-task 32 \
  --slurm-time 06:00:00 \
  --slurm-mem 512G
```

---

## Monitoring Jobs

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Detailed status of specific job
squeue -j <JOB_ID>

# Full job information
scontrol show job <JOB_ID>
```

### View Job Output

```bash
# Live output (connects to frontend first)
ssh cluster.in.fit.cvut.cz
tail -f slurm_logs/train_123456.log

# Or download logs
scp -r cluster.in.fit.cvut.cz:~/Bachelor-Thesis/slurm_logs ./
```

### Cancel Job

```bash
scancel <JOB_ID>

# Cancel all jobs
scancel -u $USER
```

### Get Full Output

```bash
# After job completes
cat slurm_logs/train_123456.log
```

---

## Common Issues

### Issue 1: "Job submission timeout"

**Symptom**: Queue is full

**Solution**: 
- Check partition availability: `sinfo`
- Submit to less-used partition or increase wait time
- Submit multiple smaller jobs instead of one large job

### Issue 2: "SSH connection refused"

**Solution**:
- Verify VPN connection
- Check SSH key: `ssh -v cluster.in.fit.cvut.cz`
- Ensure SSH key is added to GitLab profile

### Issue 3: "Out of memory (OOM) error"

**Solution**:
```bash
# Check memory usage
free -h

# Allocate more memory for next job
--slurm-mem 256G

# Or switch to larger GPU
--slurm-gpu a100_80
```

### Issue 4: "CUDA out of memory"

**Solution** (same as CPU OOM, but for GPU):
```bash
# Use GPU with more VRAM
--slurm-gpu a100_80

# Or reduce batch size in pipeline configuration
```

### Issue 5: "Module not found / Poetry not available"

**Solution**:
```bash
# Load required modules on first login
module load python/3.10
module load cuda/12.4

# Then re-run pipeline
```

### Issue 6: "Permission denied" on log directory

**Solution**:
```bash
mkdir -p ~/Bachelor-Thesis/slurm_logs
chmod 755 ~/Bachelor-Thesis/slurm_logs
```

---

## Best Practices

1. **Start with dry run** to verify configuration:
   ```bash
   --slurm-dry-run
   ```

2. **Use reasonable time limits**:
   - Planning: 2-4 hours
   - Training: 8-12 hours  
   - Inference: 1-2 hours

3. **Email notifications for long jobs**:
   ```bash
   --slurm-email your.email@fit.cvut.cz
   ```

4. **Monitor first run**: Use `--slurm-wait` for testing:
   ```bash
   --slurm-wait
   ```

5. **Save logs**: Output automatically saved to `slurm_logs/` directory

6. **GPU Scheduling**:
   - A100 is slower to queue but faster execution
   - P100/V100 queue faster but slower execution
   - Choose based on total time (queue + execution)

7. **Chain jobs**: Submit next job only after previous completes
   ```bash
   # Plan
   pipeline plan --clusterfit --slurm-wait
   # Then train
   pipeline train --clusterfit --slurm-wait
   ```

---

## Quick Reference

### Most Common Commands

```bash
# CPU Planning (fast, parallelized)
poetry run python -m src.nn_UNet.pipeline plan --clusterfit --slurm-partition fast --slurm-cpus-per-task 16

# GPU Training (A100 recommended)
poetry run python -m src.nn_UNet.pipeline train --clusterfit --slurm-partition gpu --slurm-gpu a100_40

# GPU Inference
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./images \
  --output ./predictions \
  --clusterfit \
  --slurm-partition gpu
```

---

## Additional Resources

- [ClusterFIT Documentation](https://docs.fit.cvut.cz/clusterfit/)
- [Slurm Official Docs](https://slurm.schedmd.com/)
- [nnU-Net Documentation](https://github.com/MIC-DKFZ/nnUNet)
- [FIT Support Email](mailto:support@fit.cvut.cz)

---

**Last Updated**: March 2026  
**Platform**: ClusterFIT (Slurm)  
**Python**: 3.10+  
