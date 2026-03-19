# nn_UNet ClusterFIT Quick Reference

## Quick Start

### SSH to ClusterFIT
```bash
ssh -i ~/.ssh/id_clusterfit username@cluster.in.fit.cvut.cz
```

### Run Locally
```bash
poetry run python -m src.nn_UNet.pipeline plan
poetry run python -m src.nn_UNet.pipeline train
poetry run python -m src.nn_UNet.pipeline predict --input ./images --output ./output
```

### Submit to Slurm
```bash
poetry run python -m src.nn_UNet.pipeline COMMAND --clusterfit [OPTIONS]
```

---

## Command Examples

### Planning (CPU)
```bash
# Fast multi-threaded planning
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit --slurm-partition fast --slurm-cpus-per-task 16 --slurm-time 04:00:00
```

### Training
```bash
# On GPU A100 (recommended)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit --slurm-partition gpu --slurm-gpu a100_40 --slurm-time 12:00:00

# On any GPU
poetry run python -m src.nn_UNet.pipeline train --clusterfit --slurm-partition gpu

# On CPU (slow, not recommended)
poetry run python -m src.nn_UNet.pipeline train --clusterfit --slurm-partition fast
```

### Prediction
```bash
# Quick inference on GPU
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./images --output ./predictions \
  --clusterfit --slurm-partition gpu --slurm-time 02:00:00
```

### Test Before Submitting
```bash
# Dry run (no submission)
poetry run python -m src.nn_UNet.pipeline train --clusterfit --slurm-dry-run

# Wait for completion
poetry run python -m src.nn_UNet.pipeline train --clusterfit --slurm-wait
```

---

## Slurm Flags

| Flag | Default | Example |
|------|---------|---------|
| `--clusterfit` | N/A | Enable Slurm submission |
| `--slurm-partition` | `fast` | `gpu`, `arm_fast`, `amd` |
| `--slurm-gpu` | None | `p100`, `v100`, `a100_40`, `a100_80` |
| `--slurm-time` | Auto | `02:00:00` |
| `--slurm-cpus-per-task` | `1` | `8`, `16` |
| `--slurm-mem` | Auto | `64G`, `256G` |
| `--slurm-job-name` | Auto | Custom name |
| `--slurm-email` | None | `your@email.com` |
| `--slurm-dry-run` | N/A | Test only |
| `--slurm-wait` | N/A | Block until done |

---

## GPU Comparison

| GPU | VRAM | Speed | Queue |
|---|---|---|---|
| P100 | 16GB | Slow | Fast |
| V100 | 32GB | Medium | Medium |
| A100 40GB | 40GB | Fast | Medium |
| A100 80GB | 80GB | Fast | Slow |

→ **Use A100 40GB for best balance**

---

## Partition Specs

| Name | CPU | RAM | GPU | Nodes | Best For |
|------|-----|-----|-----|-------|----------|
| `fast` | x86_64 | 64GB | None | 28 | Planning, CPU workloads |
| `gpu` | x86_64 | 64GB | NVIDIA | Multi | Training, inference |
| `arm_fast` | aarch64 | 32GB | None | 8 | ARM-specific |
| `amd` | x86_64 | 512GB | MI210 4x | 2 | Large jobs |

---

## Job Monitoring

```bash
# List all your jobs
squeue -u $USER

# Check specific job
squeue -j 12345

# Cancel job
scancel 12345

# View job info
scontrol show job 12345

# Check queue availability
sinfo
```

---

## Default Time Limits

- `plan`: 4 hours
- `train`: 12 hours
- `predict`: 2 hours
- `predict-tree`: 2 hours
- `prepare`: 1 hour

Override with: `--slurm-time HH:MM:SS`

---

## Typical Workflow

```bash
# 1. Plan (parallelized CP)
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit --slurm-partition fast --slurm-cpus-per-task 16 --slurm-wait

# 2. Train (GPU recommended)
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit --slurm-partition gpu --slurm-gpu a100_40 --slurm-wait

# 3. Predict (quick on GPU)
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./test_images --output ./results \
  --clusterfit --slurm-partition gpu
```

---

## Common Issues & Solutions

**"Out of memory"** → Add `--slurm-gpu a100_80` or `--slurm-mem 256G`

**"Queue timeout"** → Use less-busy partition or reduce job scope

**"SSH connection failed"** → Check VPN and SSH key setup

**"CUDA out of memory"** → Use larger GPU: `--slurm-gpu a100_80`

**"Module not found"** → Load modules after SSH: `module load python/3.10 cuda/12.4`

---

## Useful SSH Commands (After SSH to ClusterFIT)

```bash
# Check job logs
tail -f slurm_logs/train_123456.log

# Download all logs locally
scp -r cluster.in.fit.cvut.cz:~/Bachelor-Thesis/slurm_logs ./

# Check available modules
module avail

# Load environment
module load python/3.10
module load cuda/12.4
```

---

**Documentation**: See `CLUSTERFIT_GUIDE.md` for detailed info
