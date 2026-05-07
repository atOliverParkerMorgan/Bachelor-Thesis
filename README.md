# Wood Defect Detection in CT Scans

Bachelor thesis project for automatic wood-defect segmentation in CT scans using nnU-Net v2 and MONAI-based custom models.

**Classes:** Background (0), Healthy Wood (1), Knot (2), Rot (3), Bark (4), Crack (5), Insect Damage (6)

## Setup

```bash
poetry install
```

Place raw ground-truth data in `src/ground_truth/` as ZIP files or folders with DICOM/IMA files. The tree-specific wrappers expect to be run from the project root.

## Project Structure

```
src/
  preprocessing/          # DICOM → PNG conversion, classical segmentation, Datumaro helpers
  nn_UNet/                # nnU-Net v2 pipeline, trainer variants, cluster submission helpers
  custom_model/           # MONAI training, inference, losses, transforms, dataset code
  postprocessing/         # Rule-based cleanup and analysis utilities
    unanottated_data/       # Local raw/derived datasets (not tracked)
run                       # Classical preprocessing entrypoint
run_nnunet                # nnU-Net / custom-model entrypoint
run_mednext.sh            # Convenience wrapper for custom-train --model-name mednext
run_swinunetr.sh          # Convenience wrapper for custom-train --model-name swinunetr
```

## Workflows

### Classical preprocessing

Extract DICOM slices, convert them to PNG, run classical segmentation, and build the nnU-Net dataset:

```bash
./run
```

Useful options:

```bash
./run --masks kura,pozadi
./run --tree dub5 --skip-extract --skip-convert --masks trhlina,hniloba
./run --tree dub5 --skip-extract --skip-convert --upload
```

### nnU-Net pipeline

Prepare dataset:

```bash
./run_nnunet prepare --overwrite
```

Plan and preprocess:

```bash
./run_nnunet plan --verify-dataset-integrity
```

Train:

```bash
./run_nnunet train --configuration 3d_fullres --fold 0
```

Predict on a tree and export Datumaro:

```bash
./run_nnunet predict-tree \
    --tree DUB_4 \
    --ground-truth-root ./src/ground_truth \
    --segmentation-output-root ./predictions \
    --configuration 3d_fullres --fold 0 \
    --make-datumaro
```

Predict from a ZIP file:

```bash
./run_nnunet predict \
    --input ./src/ground_truth/DUB_4.zip \
    --output ./predictions \
    --configuration 3d_fullres --fold 0
```

### Custom models

Supported model names in `custom-train`: `swinunetr` (default), `swinunetr_v2`, `unetr`, `basicunetplusplus`, `mednext`, `segmamba`.

For the common cases, use the wrappers:

```bash
./run_swinunetr.sh --output-dir ./output/swinunetr --epochs 1000 --batch-size 2
./run_mednext.sh --output-dir ./output/mednext --epochs 1000 --batch-size 2
```

All `custom-train` flags are still available through `./run_nnunet custom-train ...` if you need a different architecture or a cluster submission.

Train with an explicit dataset split:

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

Predict with a trained custom model:

```bash
./run_nnunet custom-predict \
    --model-dir ./output/mednext \
    --input ./src/ground_truth/DUB_4.zip \
    --output ./predictions/mednext
```

Training outputs include `best_model.pth`, `last_model.pth`, `metrics_history.csv`, `training_curves.png`, and `run_summary.json`.

### Postprocessing

```bash
poetry run python -m src.postprocessing.postprocess predictions/DUB_4.nii.gz predictions/DUB_4_pp.nii.gz
poetry run python -m src.postprocessing.postprocess predictions/ predictions_postprocessed/
```

Rules applied: rot near crack becomes crack, crack near bark becomes background, small background adjacent to rot becomes rot, and enclosed healthy-wood/background holes are filled with the surrounding defect class.

## Cluster

Add `--clusterfit` and the relevant Slurm flags to any command. Recommended GPU: A100 40 GB.

### Prepare on CPU

```bash
./run_nnunet prepare --overwrite \
    --clusterfit --slurm-partition cpu \
    --slurm-cpus-per-task 16 --slurm-time 02:00:00
```

### Plan on CPU

```bash
./run_nnunet plan \
    --clusterfit --slurm-partition cpu \
    --slurm-cpus-per-task 16 --slurm-time 04:00:00 \
    --configurations 3d_fullres
```

### Train nnU-Net on GPU

```bash
./run_nnunet train \
    --clusterfit --slurm-partition gpu \
    --slurm-cpus-per-task 8 --slurm-gpu a100_40 --slurm-time 72:00:00 \
    --configuration 3d_fullres --fold 0 \
    --initial-lr 1e-3 --compile off --n-proc-da 4 --cpu-threads 1
```

### Train a custom model on GPU

```bash
./run_nnunet custom-train \
    --clusterfit --slurm-partition gpu \
    --slurm-cpus-per-task 8 --slurm-gpu a100_40 --slurm-time 24:00:00 \
    --epochs 1000 --batch-size 2 --patch-size 128 384 128 \
    --learning-rate 1e-3 --rare-class-weight 30.0 \
    --num-workers 4 --grad-accumulation-steps 4 \
    --wandb --wandb-project "bp-custom-model"
```

### Predict on GPU

```bash
./run_nnunet predict \
    --clusterfit --slurm-partition gpu --slurm-gpu a100_40 \
    --input ./src/ground_truth/DUB_4.zip \
    --output ./predictions \
    --configuration 3d_fullres --fold 0
```

## Useful Commands

```bash
poetry run python src/nn_UNet/label_stats.py --csv stats.csv
poetry run python src/custom_model/visualize_augmentation.py
poetry run python src/preprocessing/utils/zorder_cvat_fix.py --tree dub4
poetry run python src/preprocessing/conversion/predict2datumaro.py --tree DUB_4
```
