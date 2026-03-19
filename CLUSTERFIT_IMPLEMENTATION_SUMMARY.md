# ClusterFIT Integration - Implementation Summary

**Date**: March 19, 2026  
**Project**: Bachelor-Thesis nn_UNet Pipeline  
**Platform**: ClusterFIT (Slurm Job Scheduler)

## Objective ✅ COMPLETED

Fix the nn_UNet pipeline implementation to run correctly and efficiently on ClusterFIT, including:
- Slurm job submission support
- GPU and CPU resource selection
- Efficient resource allocation
- Comprehensive documentation

---

## Deliverables

### 1. New Modules & Utilities

#### `src/nn_UNet/clusterfit_utils.py` (350 lines)
**Purpose**: Core Slurm submission framework

**Key Components**:
- `SlurmConfig` dataclass - Configuration for job parameters
- `SlurmJobSubmitter` - Main class for job management
- `add_clusterfit_arguments()` - Argument parser integration
- `build_slurm_config_from_args()` - Configuration builder from CLI args
- `build_slurm_script()` - Bash script generation
- GPU model validation (p100, v100, a100_40, a100_80, mi210)

**Features**:
✓ SBATCH directive generation  
✓ Environment variable forwarding  
✓ Module loading for ARM/HPE CPE  
✓ Automatic time limit defaults  
✓ Job submission and monitoring  

---

### 2. Slurm Job Templates

**Location**: `src/nn_UNet/slurm_templates/`

| Script | Purpose | Resources |
|--------|---------|-----------|
| `plan_cpu.sh` | Preprocessing (planning) | 8 CPU cores |
| `train_gpu_a100.sh` | Training on Tesla A100 40GB | 1x Tesla A100 40GB |
| `train_gpu_p100.sh` | Training on Tesla P100 16GB | 1x Tesla P100 16GB |
| `train_arm_hpe.sh` | Training on ARM with HPE modules | Fujitsu A64FX |
| `predict_gpu.sh` | Inference | 1x GPU (any model) |

**Features**:
- SBATCH directives for resource allocation
- Module loading for GPU/ARM environments
- Environment variable setup
- Job logging and monitoring info
- Ready-to-use examples

---

### 3. Enhanced Pipeline

**Modified**: `src/nn_UNet/pipeline.py`

**Changes**:
1. **Imports**: Added ClusterFIT utilities
   ```python
   from src.nn_UNet.clusterfit_utils import (
       SlurmJobSubmitter, SlurmConfig, build_slurm_config_from_args,
       add_clusterfit_arguments,
   )
   ```

2. **Argument Parsers**: Added `--clusterfit` group to:
   - `plan` subcommand
   - `train` subcommand
   - `predict` subcommand
   - `predict-tree` subcommand
   - `all` subcommand

3. **New Function**: `submit_to_clusterfit()`
   - Rebuilds command with all arguments
   - Generates Slurm batch script
   - Submits to job queue
   - Returns job ID

4. **Main Function**: Enhanced with ClusterFIT check
   ```python
   if getattr(args, "clusterfit", False):
       submit_to_clusterfit(args, env)
       return
   ```

---

### 4. Documentation

#### `CLUSTERFIT_GUIDE.md` (400+ lines)
**Comprehensive User Guide**

Sections:
- ✓ System architecture overview
- ✓ Authentication setup (SSH keys, VPN)
- ✓ GPU selection guide with recommendations
- ✓ CPU architecture selection (x86_64 vs aarch64)
- ✓ 5 detailed examples
- ✓ Job monitoring commands
- ✓ 6 common issues with solutions
- ✓ Best practices
- ✓ Quick reference commands
- ✓ Additional resources

#### `CLUSTERFIT_QUICK_REF.md` (150+ lines)
**Quick Reference Card**

Contents:
- Instant commands for common tasks
- GPU comparison table
- Partition specifications
- Slurm flags reference
- Default time limits
- Typical workflow
- Issue solutions

#### `README_CLUSTERFIT.md` (200+ lines)
**Implementation Overview**

Contents:
- Files added/modified summary
- Quick start guide
- Core concepts
- Common workflows
- Monitoring instructions
- Example commands
- Troubleshooting

---

## Supported Features

### Resource Selection

**Partitions**:
- `fast` - CPU nodes (28x, 64GB RAM, 36 cores)
- `gpu` - GPU nodes (NVIDIA P100/V100/A100)
- `arm_fast` - ARM nodes (8x, aarch64, 32GB RAM)
- `amd` - Large GPU nodes (2x, 512GB RAM, 4x MI210)

**GPUs**:
- NVIDIA Tesla P100 (16GB) - fast queuing
- NVIDIA Tesla V100 (32GB) - balanced
- NVIDIA Tesla A100 (40GB) - **RECOMMENDED**
- NVIDIA Tesla A100 (80GB) - large models
- AMD Instinct MI210 (4x per node) - experimental

**CPUs**:
- x86_64 (Intel Xeon) - default
- aarch64 (Fujitsu A64FX) - ARM support with HPE CPE

### Command Arguments

All pipeline commands support:
```
--clusterfit                  # Enable Slurm submission
--slurm-partition             # Partition: fast, gpu, arm_fast, amd
--slurm-gpu                   # GPU model: p100, v100, a100_40, etc.
--slurm-time HH:MM:SS        # Time limit
--slurm-cpus-per-task N      # CPU cores for parallelization
--slurm-mem SIZE             # Memory override (e.g., 256G)
--slurm-nodes N              # Multi-node jobs
--slurm-job-name NAME        # Custom job name
--slurm-output PATH          # Log file path
--slurm-email EMAIL          # Email notifications
--slurm-dry-run              # Test without submitting
--slurm-wait                 # Wait for completion
--arm-hpe-cpe                # Load HPE modules for ARM
```

### Default Time Limits

- **plan**: 4 hours
- **train**: 12 hours
- **predict**: 2 hours
- **predict-tree**: 2 hours
- **prepare**: 1 hour

---

## Example Usage

### Planning (CPU Parallelized)
```bash
poetry run python -m src.nn_UNet.pipeline plan \
  --clusterfit \
  --slurm-partition fast \
  --slurm-cpus-per-task 16 \
  --slurm-time 04:00:00
```

### Training (GPU Recommended)
```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-partition gpu \
  --slurm-gpu a100_40 \
  --slurm-time 12:00:00
```

### Inference (Fast GPU)
```bash
poetry run python -m src.nn_UNet.pipeline predict \
  --input ./images \
  --output ./predictions \
  --clusterfit \
  --slurm-partition gpu
```

### Test Submission
```bash
poetry run python -m src.nn_UNet.pipeline train \
  --clusterfit \
  --slurm-dry-run
```

---

## Technical Implementation

### Job Submission Flow

```
1. User runs: poetry run python -m src.nn_UNet.pipeline COMMAND --clusterfit
2. Arguments parsed and validated
3. SlurmConfig created from args
4. Original command reconstructed with all parameters
5. Slurm batch script generated
6. Script written to temporary file
7. Script made executable
8. sbatch submitted
9. Job ID returned to user
10. Logs available in slurm_logs/
```

### Script Generation

The pipeline automatically:
- ✓ Reconstructs command with ALL arguments
- ✓ Generates SBATCH directives
- ✓ Forwards environment variables
- ✓ Loads necessary modules
- ✓ Handles GPU model selection
- ✓ Supports ARM/HPE CPE requirements
- ✓ Creates log directory structure

### Error Handling

- ✅ GPU model validation
- ✅ Script path verification
- ✅ File permission handling
- ✅ Command availability checks
- ✅ Temp file cleanup

---

## Testing Checklist

- ✅ Pipeline.py imports correctly
- ✅ No syntax errors in clusterfit_utils.py
- ✅ All argument parsers accept ClusterFIT flags
- ✅ CLI argument reconstruction works
- ✅ Slurm script generation produces valid bash
- ✅ Job submission function handles all commands
- ✅ Dry run mode works without submitting
- ✅ GPU model validation functions
- ✅ Documentation is complete and clear

---

## Recommendations for Use

1. **Start Simple**: Try `--slurm-dry-run` first to see generated script
2. **Use Wait Mode**: `--slurm-wait` for initial testing
3. **A100 Recommended**: Best speed/availability balance for nnU-Net
4. **Parallelize Planning**: Use 8-16 CPU cores for preprocessing
5. **Monitor Early**: Check first job logs for issues
6. **Set Notifications**: Use `--slurm-email` for long jobs

---

## Performance Expectations

| Stage | Partition | GPU | Typical Duration |
|-------|-----------|-----|------------------|
| Plan | fast (16 cores) | None | 30 minutes - 2 hours |
| Train | gpu | A100 40GB | 4-12 hours |
| Predict | gpu | Any | 5-30 minutes |

---

## Future Enhancements (Optional)

- [ ] Multi-node MPI support for distributed training
- [ ] Checkpoint restart from previous jobs
- [ ] Job chaining (auto-submit next stage)
- [ ] Resource usage monitoring dashboard
- [ ] AMD MI210 GPU support testing
- [ ] Cost estimation before submission

---

## Files Summary

```
src/nn_UNet/
├── clusterfit_utils.py              # ✅ Slurm utilities (350 lines)
├── pipeline.py                       # ✅ Updated with ClusterFIT support
├── README_CLUSTERFIT.md             # ✅ Implementation overview
├── CLUSTERFIT_GUIDE.md              # ✅ Comprehensive guide (400+ lines)
├── CLUSTERFIT_QUICK_REF.md          # ✅ Quick reference
└── slurm_templates/                 # ✅ Example scripts
    ├── plan_cpu.sh
    ├── train_gpu_a100.sh
    ├── train_gpu_p100.sh
    ├── train_arm_hpe.sh
    └── predict_gpu.sh
```

---

## Verification Commands

```bash
# Test imports
python -c "from src.nn_UNet.clusterfit_utils import SlurmJobSubmitter; print('✓ Imports OK')"

# Test pipeline help
poetry run python -m src.nn_UNet.pipeline train --help | grep slurm

# List templates
ls -la src/nn_UNet/slurm_templates/

# Check documentation
wc -l src/nn_UNet/CLUSTERFIT*.md
```

---

## Status: ✅ COMPLETE

All objectives achieved:
- ✅ ClusterFIT utilities implemented
- ✅ Job templates created
- ✅ Pipeline integrated with Slurm support
- ✅ Comprehensive documentation provided
- ✅ GPU/CPU resource selection enabled
- ✅ Ready for production use

The nn_UNet pipeline is now fully compatible with ClusterFIT and can efficiently utilize its computational resources for planning, training, and inference.

---

**Date Completed**: March 19, 2026
