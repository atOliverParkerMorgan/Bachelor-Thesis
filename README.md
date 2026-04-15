# BP

Simple workflow for wood-defect segmentation and nnU-Net training.

## 1) Where to put data

### Ground truth for extraction and segmentation pipeline

Put your source zip files here:

- src/ground_truth

Example:

```text
src/ground_truth/
	dub1.zip
	dub2.zip
	dub5.zip
```

### Data for nnU-Net training input

Put extracted per-tree folders here:


Expected structure per tree:

```text
	nn_Unent/dataset/dub1/
		labelmap.txt
		SegmentationObject/*.png
		SegmentationClass/*.png
		ImageSets/Segmentation/dub1.txt   (optional)
```

### Where nnU-Net generated data is stored

The pipeline now uses:

- src/nn_Unet/nnunet_data/nnUNet_raw
- src/nn_Unet/nnunet_data/nnUNet_preprocessed
- src/nn_Unet/nnunet_data/nnUNet_results

## 2) Install

```bash
poetry install
```

## 3) Segmentation commands (keep this)

### Full segmentation pipeline

```bash
./run
```

### Segmentation for one tree

```bash
poetry run python src/preprocessing/segmentation/segmentation.py --tree dub1
```

### Segmentation for one tree with selected masks

```bash
poetry run python src/preprocessing/segmentation/segmentation.py --tree dub1 --masks kura trhlina
```

### Useful run options

```bash
./run --masks kura,pozadi
./run --skip-extract --skip-convert --masks trhlina,hniloba
./run --tree dub5 --skip-extract --skip-convert --upload --upload-job-id 12345 --masks pozadi

# predictions
./run --tree dub4 --clean-logs-only
```

## 4) nnU-Net on your computer (local)

### Step A: prepare Dataset001 to nnU-Net format

```bash
./run_nnunet prepare --overwrite
```

### Step B: plan and preprocess

```bash
./run_nnunet plan --verify-dataset-integrity
```

For 3d_lower profile:

```bash
./run_nnunet plan --configurations 3d_lower --num-processes 1
```

### Step C: train

3d_fullres:

```bash
./run_nnunet train --configuration 3d_fullres --fold 0
```

3d_lower:

```bash
./run_nnunet train --configuration 3d_lower --fold 0
```

Resume training:

```bash
./run_nnunet train --configuration 3d_lower --fold 0 --continue-training
```

## 5) nnU-Net on ClusterFIT

Use the same commands with cluster flags.

### Prepare
```bash
./run_nnunet prepare --overwrite --clusterfit --slurm-partition cpu --slurm-cpus-per-task 16 --slurm-time 02:00:00
```

### Plan on CPU queue

```bash
./run_nnunet plan \
	--clusterfit \
	--slurm-partition cpu \
	--slurm-cpus-per-task 16 \
	--slurm-time 04:00:00 \
	--configurations 3d_fullres
```

### Train on GPU queue (recommended A100 40GB)

```bash
./run_nnunet train \
	--clusterfit \
	--slurm-partition gpu \
	--slurm-cpus-per-task 8 \
	--slurm-gpu a100_40 \
	--slurm-time 72:00:00 \
	--configuration 3d_fullres \
	--compile off \
	--n-proc-da 4 \
	--cpu-threads 1 \
	--fold 0
	
```

If training appears stuck before epoch logs on `3d_fullres`, use this safer command:

```bash
./run_nnunet train \
	--clusterfit \
	--slurm-partition gpu \
	--slurm-cpus-per-task 8 \
	--slurm-gpu a100_40 \
	--slurm-time 24:00:00 \
	--configuration 3d_fullres \
	--compile off \
	--n-proc-da 4 \
	--cpu-threads 1 \
	--fold 0
```

### Transfer learning with pretrained Lung weights (recommended)

Lung CT and wood CT share similar HU value ranges, making the Lung model the best
pretrained starting point.  The custom trainer `nnUNetTrainerLungPretrained` loads
encoder weights with `strict=False` so the output head always trains from scratch.

**One-time setup — run on the cluster before training:**

```bash
# Upload the zip to the cluster first, then:
nnUNetv2_install_pretrained_model_from_zip Task006_Lung.zip
# Prints the installed results path, e.g.:
# $nnUNet_results/Dataset006_Lung/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth
```

**Get dataset statistics**
```bash
poetry run python src/nn_UNet/label_stats.py ./src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/labelsTr  --csv stats.csv
```

**Train with pretrained weights and W&B logging (best parameters for this dataset):**


```bash
./run_nnunet train \
    --clusterfit \
    --slurm-partition gpu \
    --slurm-cpus-per-task 8 \
    --slurm-gpu a100_40 \
    --slurm-time 48:00:00 \
    --configuration 3d_fullres \
    --fold 0 \
    --compile off \
    --n-proc-da 4 \
    --cpu-threads 1 \
    --initial-lr 1e-3 \
    --pretrained-weights $nnUNet_results/3d_fullres/Task006_Lung/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_final_checkpoint.model \
    --wandb \
    --wandb-project "bp-wood-defects" \
    --wandb-entity "oliver-parker-morgan-czech-technical-university-in-prague"
```


**Why these parameters:**

| Parameter | Value | Reason |
|---|---|---|
| `--configuration` | `3d_fullres` | Best accuracy for 3-D wood CT volumes |
| `--initial-lr` | `1e-3` | Lower than default (0.01); prevents overwriting pretrained features early |
| `--n-proc-da` | `4` | Saturates A100 40 GB without memory pressure |
| `--compile` | `off` | Avoids torch.compile startup hang on the cluster |
| `--slurm-time` | `48:00:00` | Sufficient for 1000 epochs on a small dataset with transfer weights |
| `--fold` | `0` | Single fold; add `1 2 3 4` as separate jobs once fold 0 validates |

The pipeline automatically:
1. Copies `nnunet_trainer_pretrained.py` into the nnunetv2 variants directory so
   `nnUNetv2_train -tr nnUNetTrainerLungPretrained` can discover it.
2. Passes the checkpoint path via `NNUNET_PRETRAINED_WEIGHTS`.
3. Falls back to nnUNetv2's built-in `--pretrained_weights` if the copy fails.

## 6) Custom 3D model (SwinUNETR)

The custom trainer uses a 3-D **SwinUNETR** (Swin Transformer U-Net) from MONAI.
Label 6 (`poskozeni_hmyzem`) is boosted with the same three-layer strategy used by
`nnUNetTrainerRareClassBoostWandb`:

1. **Case-level oversampling** — cases containing label 6 are duplicated 8× so they appear more often per epoch.
2. **Patch-level focus** — `RandCropByLabelClassesd` gives label-6 voxels 8× higher crop-centre probability.
3. **Weighted CE loss** — label-6 voxels contribute 30× more to the cross-entropy term of the DiceCE loss.

| Setting | Value |
|---|---|
| Model | SwinUNETR (`feature_size=48`) |
| Patch size | `128 × 384 × 128` |
| Batch size (crops / volume) | `2` |
| Optimizer | AdamW |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | Cosine annealing |
| Epochs | `1000` |
| Classes | `7` |
| Intensity clipping | `[−1000, 3000] → [0, 1]` |
| Rare-class oversample | `8×` case-level |
| Rare-class CE weight | `30×` (label 6) |

### Local run

```bash
poetry run python -m src.custom_model.train \
	--image-dir ./src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/imagesTr \
	--label-dir ./src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/labelsTr \
	--output-dir ./output/custom_model \
	--epochs 1000 \
	--batch-size 2 \
	--patch-size 128 384 128 \
	--learning-rate 1e-3 \
	--num-classes 7 \
	--val-fraction 0.25 \
	--wandb \
	--wandb-project "bp-custom-model"
```

### Run on ClusterFIT (recommended)

Uses `./run_nnunet custom-train`. Image/label directories default to
`nnunet_root/nnUNet_raw/Dataset001_BPWoodDefects/{imagesTr,labelsTr}`.

```bash
./run_nnunet custom-train \
    --clusterfit \
    --slurm-partition gpu \
    --slurm-cpus-per-task 8 \
    --slurm-gpu a100_40 \
    --slurm-time 24:00:00 \
    --epochs 1000 \
    --batch-size 2 \
    --patch-size 128 384 128 \
	--learning-rate 1e-3 \
	--wandb \
	--wandb-project "bp-custom-model"
```

Resume from a checkpoint is not currently supported by the custom model — restart training with `--output-dir` pointing to a fresh directory or keep `last_model.pth` and re-implement resumption manually.

The trainer writes `best_model.pth`, `last_model.pth`, and the resolved config JSON into the output directory.

### Predict on cluster

```bash
./run_nnunet predict \
	--clusterfit \
	--slurm-partition gpu \
	--slurm-gpu a100_40 \
	--input ./src/ground_truth/DUB_4.zip \
	--output ./predictions \
	--configuration 3d_fullres \
	--fold 0

./run_nnunet predict \
--clusterfit \
--slurm-partition gpu \
--slurm-gpu a100_40 \
--input DUB_4.zip \
--output ./predictions \
--configuration 3d_fullres \
--fold 0 \
--trainer nnUNetTrainerLungPretrainedWandb \
--plans-identifier nnUNetResEncUNetLPlans \

# full pipeline to -datumaro	
./run_nnunet predict-tree \
    --clusterfit \
    --slurm-partition gpu \
    --slurm-gpu a100_40 \
    --tree DUB_4 \
    --ground-truth-root ./src/ground_truth \
	--segmentation-output-root ./predictions \
    --configuration 3d_fullres \
    --fold 0 \
    --make-datumaro	

# preparation for fixing cvat trees
# run this if the tree doesnt exist
./run --tree dub4 --masks kura,pozadi

poetry run python src/preprocessing/utils/zorder_cvat_fix.py --tree dub4

./run_nnunet predict-tree \
    --clusterfit \
    --slurm-partition gpu \
    --tree DUB_4 \
    --ground-truth-root ./src/ground_truth \
    --segmentation-output-root ./src/nn_UNet/predictions \
    --configuration 3d_fullres \
    --fold 0 \
    --make-datumaro

# to a cvat format
poetry run python src/preprocessing/conversion/nii2mask.py --tree dub_4
```

## 7) Useful project files

- [run](run)
- [run_nnunet](run_nnunet)
- [src/nn_UNet/pipeline.py](src/nn_UNet/pipeline.py)
- [src/preprocessing/segmentation/segmentation.py](src/preprocessing/segmentation/segmentation.py)
- [src/preprocessing/conversion/segmentmask2nnunetformat.py](src/preprocessing/conversion/segmentmask2nnunetformat.py)

## Notes

- If you only have zip files for segmentation workflow, place them in src/ground_truth and use ./run.
- For nnU-Net training, data must be extracted into nn_Unet/datasets/Dataset001/dubX folders.
- If geometry exists, keep geometry.json under src/png/dubX for best spacing metadata.

