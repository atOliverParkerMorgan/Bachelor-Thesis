# Wood Defect Detection in CT Scans

Bachelor thesis project — automatic segmentation of wood defects in CT scans using nnU-Net v2 and a custom SwinUNETR model.

**Classes:** Background (0), Healthy Wood (1), Knot (2), Rot (3), Bark (4), Crack (5), Insect Damage (6)

---

## Setup

```bash
poetry install
```

Place raw ground truth data (ZIP files or folders with DICOM/IMA files) in `src/ground_truth/`.

---

## Project structure

```
src/
  preprocessing/          # DICOM → PNG → classical segmentation masks → nnU-Net format
  nn_UNet/                # nnU-Net v2 pipeline wrapper + custom trainer variants
  custom_model/           # SwinUNETR / MedNexT training pipeline
  postprocessing/         # Rule-based prediction cleanup
src/ground_truth/         # raw data — not tracked
src/nn_UNet/nnunet_data/  # nnU-Net data — not tracked
output/                   # model checkpoints and results — not tracked
```

---

## Workflows

### 1. Classical preprocessing

Extracts DICOM slices, converts to PNG, runs classical segmentation, and builds the nnU-Net dataset:

```bash
./run
```

Useful options:

```bash
./run --masks kura,pozadi
./run --tree dub5 --skip-extract --skip-convert --masks trhlina,hniloba
./run --tree dub5 --skip-extract --skip-convert --upload
```

---

### 2. nnU-Net pipeline

#### Prepare dataset (convert to nnU-Net format)

```bash
./run_nnunet prepare --overwrite
```

#### Plan and preprocess

```bash
./run_nnunet plan --verify-dataset-integrity
```

#### Train

```bash
./run_nnunet train --configuration 3d_fullres --fold 0
```

#### Predict on a tree (NIfTI input → masks → Datumaro ZIP)

```bash
./run_nnunet predict-tree \
    --tree DUB_4 \
    --ground-truth-root ./src/ground_truth \
    --segmentation-output-root ./predictions \
    --configuration 3d_fullres --fold 0 \
    --make-datumaro
```

#### Predict from a ZIP

```bash
./run_nnunet predict \
    --input ./src/ground_truth/DUB_4.zip \
    --output ./predictions \
    --configuration 3d_fullres --fold 0
```

---

### 3. Custom model (SwinUNETR / MedNeXt)

Available models: `swinunetr` (default), `swinunetr_v2`, `mednext`, `segmamba`, `unetr`, `basicunetplusplus`.

#### Train locally

```bash
./run_nnunet custom-train \
    --model-name mednext \
    --image-dir ./src/nn_UNet/nnunet_data/nnUNet_raw/Dataset002_BPWoodDefectsSplit/imagesTr \
    --label-dir ./src/nn_UNet/nnunet_data/nnUNet_raw/Dataset002_BPWoodDefectsSplit/labelsTr \
    --output-dir ./output/mednext \
    --epochs 1000 --batch-size 2 --patch-size 128 384 128 \
    --learning-rate 1e-3 --rare-class-weight 15.0 \
    --num-workers 4 --grad-accumulation-steps 4 \
    --wandb --wandb-project "bp-custom-model"
```

Resume from a checkpoint:

```bash
./run_nnunet custom-train ... --resume-checkpoint ./output/mednext/last_model.pth
```

#### Train on cluster (GPU)

```bash
./run_nnunet custom-train \
    --model-name mednext \
    --output-dir ./output/mednext \
    --epochs 1000 --batch-size 2 --patch-size 128 384 128 \
    --learning-rate 1e-3 --rare-class-weight 15.0 \
    --num-workers 4 --grad-accumulation-steps 4 \
    --wandb --wandb-project "bp-custom-model" \
    --clusterfit --slurm-partition gpu \
    --slurm-cpus-per-task 8 --slurm-gpu a100_40 --slurm-time 24:00:00
```

#### Predict

```bash
./run_nnunet custom-predict \
    --model-dir ./output/mednext \
    --input ./src/ground_truth/DUB_4.zip \
    --output ./predictions/mednext
```

Outputs written to `--output-dir`: `best_model.pth`, `last_model.pth`, `metrics_history.csv`, `training_curves.png`, `run_summary.json`.

---

### 4. Postprocessing

```bash
# single file
poetry run python -m src.postprocessing.postprocess predictions/DUB_4.nii.gz predictions/DUB_4_pp.nii.gz

# whole directory
poetry run python -m src.postprocessing.postprocess predictions/ predictions_postprocessed/
```

Rules applied: rot near crack → crack, crack near bark → background, small background adjacent to rot → rot, enclosed HW/BG holes → filled with surrounding defect.

---

## Cluster (ClusterFIT / Slurm)

Add `--clusterfit` and Slurm flags to any command. Recommended GPU: A100 40 GB.

### Prepare (CPU)

```bash
./run_nnunet prepare --overwrite \
    --clusterfit --slurm-partition cpu \
    --slurm-cpus-per-task 16 --slurm-time 02:00:00
```

### Plan (CPU)

```bash
./run_nnunet plan \
    --clusterfit --slurm-partition cpu \
    --slurm-cpus-per-task 16 --slurm-time 04:00:00 \
    --configurations 3d_fullres
```

### Train nnU-Net (GPU)

```bash
./run_nnunet train \
    --clusterfit --slurm-partition gpu \
    --slurm-cpus-per-task 8 --slurm-gpu a100_40 --slurm-time 72:00:00 \
    --configuration 3d_fullres --fold 0 \
    --initial-lr 1e-3 --compile off --n-proc-da 4 --cpu-threads 1
```

### Train custom model (GPU)

```bash
./run_nnunet custom-train \
    --clusterfit --slurm-partition gpu \
    --slurm-cpus-per-task 8 --slurm-gpu a100_40 --slurm-time 24:00:00 \
    --epochs 1000 --batch-size 2 --patch-size 128 384 128 \
    --learning-rate 1e-3 --rare-class-weight 30.0 \
    --num-workers 4 --grad-accumulation-steps 4 \
    --wandb --wandb-project "bp-custom-model"
```

### Predict (GPU)

```bash
./run_nnunet predict \
    --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
    --input ./src/ground_truth/DUB_4.zip \
    --output ./predictions \
    --configuration 3d_fullres --fold 0
```

---

## Useful one-liners

```bash
# Dataset label statistics
poetry run python src/nn_UNet/label_stats.py --csv stats.csv

# Preview augmentations (writes NIfTI patches for 3D Slicer)
poetry run python src/custom_model/visualize_augmentation.py

# Fix CVAT z-order for a tree
poetry run python src/preprocessing/utils/zorder_cvat_fix.py --tree dub4

# NIfTI predictions → Datumaro ZIP (when you already have .nii.gz predictions)
poetry run python src/preprocessing/conversion/predict2datumaro.py --tree DUB_4
```
