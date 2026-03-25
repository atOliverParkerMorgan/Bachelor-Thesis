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

- datasets/Dataset001

Expected structure per tree:

```text
datasets/Dataset001/
	dub1/
		labelmap.txt
		SegmentationObject/*.png
		SegmentationClass/*.png
		ImageSets/Segmentation/dub1.txt   (optional)
	dub2/
	dub5/
```

### Where nnU-Net generated data is stored

The pipeline now uses:

- datasets/nnunet_data/nnUNet_raw
- datasets/nnunet_data/nnUNet_preprocessed
- datasets/nnunet_data/nnUNet_results

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
	--configurations 3d_lower
```

### Train on GPU queue (recommended A100 40GB)

```bash
./run_nnunet train \
	--clusterfit \
	--slurm-partition gpu \
	--slurm-cpus-per-task 8 \
	--slurm-gpu a100_40 \
	--slurm-time 24:00:00 \
	--configuration 3d_lower \
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

### Predict on cluster

```bash
./run_nnunet predict \
	--clusterfit \
	--slurm-partition gpu \
	--slurm-gpu a100_40 \
	--input ./predictions/src \
	--output ./predictions/out \
	--configuration 3d_fulres \
	--fold 0
```

### Monitor jobs

```bash
squeue -u $USER
squeue -j JOB_ID
tail -f slurm_logs/*.log
```

Monitor:
```bash
squeue -u $USER
tail -f slurm_logs/nnunet-plan_JOBID.log
```
## 6) Predict one tree and export/upload

```bash
./run_nnunet predict-tree --tree dub5 --configuration 2d --fold 0 --make-datumaro
./run_nnunet predict-tree --tree dub5 --configuration 2d --fold 0 --upload-cvat
```

## 7) Useful project files

- [run](run)
- [run_nnunet](run_nnunet)
- [src/nn_UNet/pipeline.py](src/nn_UNet/pipeline.py)
- [src/preprocessing/segmentation/segmentation.py](src/preprocessing/segmentation/segmentation.py)
- [src/preprocessing/conversion/segmentmask2nnunetformat.py](src/preprocessing/conversion/segmentmask2nnunetformat.py)

## Notes

- If you only have zip files for segmentation workflow, place them in src/ground_truth and use ./run.
- For nnU-Net training, data must be extracted into datasets/Dataset001/dubX folders.
- If geometry exists, keep geometry.json under src/png/dubX for best spacing metadata.

### Predict one tree from Datumaro zip using 3d_fullres

```bash
# 1. Unzip the Datumaro dataset
unzip predict-data/datumaro_dub4.zip -d predict-data/dub4

# 2. Run nnU-Net prediction for dub4 using 3d_fullres model
./run_nnunet predict-tree --tree dub4 --input-root predict-data --configuration 3d_fullres --fold 0 --make-datumaro
```

