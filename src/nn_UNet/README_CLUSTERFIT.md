# nn_UNet Pipeline - ClusterFIT Support

This directory now includes complete ClusterFIT (Slurm) integration for running nn_UNet training, planning, and inference on FIT ČVUT's high-performance computing platform.

## Files Added/Modified

### New Files

| File | Purpose |
|------|---------|
| [clusterfit_utils.py](clusterfit_utils.py) | Slurm submission utilities and configuration |
| [slurm_templates/plan_cpu.sh](slurm_templates/plan_cpu.sh) | Example: CPU planning job |
| [slurm_templates/train_gpu_a100.sh](slurm_templates/train_gpu_a100.sh) | Example: GPU training (A100 40GB) |
| [slurm_templates/train_gpu_p100.sh](slurm_templates/train_gpu_p100.sh) | Example: GPU training (P100 16GB) |
| [slurm_templates/train_arm_hpe.sh](slurm_templates/train_arm_hpe.sh) | Example: ARM training with HPE modules |
| [slurm_templates/predict_gpu.sh](slurm_templates/predict_gpu.sh) | Example: GPU inference |
| [CLUSTERFIT_GUIDE.md](CLUSTERFIT_GUIDE.md) | **📖 Comprehensive guide (START HERE)** |
| [CLUSTERFIT_QUICK_REF.md](CLUSTERFIT_QUICK_REF.md) | Quick reference with common commands |

### Modified Files

| File | Changes |
|------|---------|
| [pipeline.py](pipeline.py) | Added ClusterFIT arguments and submission logic |

## Quick Start

### 1. Connect to ClusterFIT

```bash
ssh -i ~/.ssh/id_clusterfit username@cluster.in.fit.cvut.cz
```

### 2. Navigate to Project

```bash
cd ~/Bachelor-Thesis
```

### 3. Submit Job to Slurm

```bash
# Planning on CPU (parallelized)
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00

# Training on GPU (A100 recommended)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 12:00:00

# Inference
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./images \
  --output ./predictions \
  --clusterfit
```

## Core Concepts

### Partitions

- **`fast`** - CPU nodes (64GB RAM, 36 cores), good for planning/preprocessing
- **`gpu`** - GPU nodes (NVIDIA P100/V100/A100), for training/inference  
- **`arm_fast`** - ARM nodes (aarch64, 32GB RAM), for ARM-specific workloads
- **`amd`** - Large GPU nodes (512GB RAM, 4x AMD MI210), for huge jobs

### GPU Options

- **`p100`** - 16GB, slower, queues fast
- **`v100`** - 32GB, medium speed
- **`a100_40`** - 40GB, fast **(recommended for nnU-Net)**
- **`a100_80`** - 80GB, very fast but rare
- **`mi210`** - AMD GPU, experimental

### Main Flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--clusterfit` | Submit to Slurm instead of running locally | `--clusterfit` |
| `--slurm-partition` | Compute partition | `fast`, `gpu`, `arm_fast` |
| `--slurm-gpu` | Specific GPU model | `a100_40`, `v100`, `p100` |
| `--slurm-time` | Time limit | `04:00:00`, `12:00:00` |
| `--slurm-cpus-per-task` | CPU cores for parallelization | `8`, `16` |
| `--slurm-dry-run` | Test without submitting | `--slurm-dry-run` |
| `--slurm-wait` | Wait for job completion | `--slurm-wait` |

## Common Workflows

### Typical Full Pipeline

```bash
# 1. Plan (CPU, 4 hours, parallelized)
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00 \
  --slurm-wait

# 2. Train (GPU, 12 hours, A100 recommended)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 12:00:00 \
  --slurm-wait

# 3. Predict (GPU, 2 hours, fast)
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_data \
  --output ./results \
  --clusterfit \
  --slurm-partition gpu
```

### Monitor Jobs

```bash
# After SSH to ClusterFIT

# List all your jobs
squeue -u $USER

# Check specific job
squeue -j 12345

# View job details
scontrol show job 12345

# Cancel job
scancel 12345

# Check queue availability
sinfo
```

## Documentation

- **[CLUSTERFIT_GUIDE.md](CLUSTERFIT_GUIDE.md)** - Detailed guide with architecture overview, authentication, GPU selection, troubleshooting
- **[CLUSTERFIT_QUICK_REF.md](CLUSTERFIT_QUICK_REF.md)** - Quick commands and reference tables
- **[slurm_templates/](slurm_templates/)** - Example batch scripts for different scenarios

## Key Improvements

✅ **Efficient Resource Usage**
- Plan on CPU (16 cores) - parallelized preprocessing
- Train on high-end GPU (A100) - faster convergence
- Automatic GPU VRAM detection for optimal inference parameters

✅ **Flexible Job Submission**
- Submit to different partitions based on workload type
- GPU model selection for cost/speed tradeoff
- Time limit override for long-running jobs

✅ **Better Monitoring**
- Job IDs returned after submission
- Log files saved to `slurm_logs/` directory
- Email notifications support

✅ **Production Ready**
- Dry run mode for testing configurations
- Environment variable forwarding
- HPE Cray modules for ARM architecture
- Automatic script generation and submission

## Default Time Limits

Following defaults are set if not specified:

- `plan` → 4 hours
- `train` → 12 hours
- `predict` → 2 hours
- `predict-tree` → 2 hours

Override with `--slurm-time HH:MM:SS`

## Example: Test Dry Run

Before submitting for real, test your command:

```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-gpu a100_40 \
  --slurm-dry-run
```

This will show the Slurm script without submitting.

## Next Steps

1. 📖 Read [CLUSTERFIT_GUIDE.md](CLUSTERFIT_GUIDE.md) for detailed information
2. 🚀 Try a simple command: `poetry run python -m src.nn_UNet.pipeline plan --clusterfit --slurm-dry-run`
3. 🎯 Run your first job with `--slurm-wait` to monitor it
4. 📊 Check logs in `slurm_logs/` directory

## Troubleshooting

**Connection issues?**
- Ensure VPN is active
- Check SSH key is added to GitLab profile  
- Verify SSH config: `ssh -v cluster.in.fit.cvut.cz`

**Job stuck in queue?**
- Use faster-queuing GPU: `--slurm-gpu p100`
- Submit to less-busy partition
- Reduce time requirement

**Out of memory?**
- Use larger GPU: `--slurm-gpu a100_80`
- Allocate more memory: `--slurm-mem 256G`
- Use AMD partition: `--slurm-partition amd`

**Module not found?**
- Load after SSH: `module load python/3.10 cuda/12.4`
- Setup once in `.bashrc` or `.bash_profile`

For more help, see [CLUSTERFIT_GUIDE.md](CLUSTERFIT_GUIDE.md) "Common Issues" section.

---

**Platform**: FIT ČVUT ClusterFIT (Slurm scheduler)  
**Updated**: March 2026  
**Implementation**: Full integration with automatic script generation and submission
