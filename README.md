# BP

Utilities for converting DICOM/IMA series to PNG slices (with geometry metadata) and back to 3D NIfTI.

## Project structure

- [src/preprocessing/conversion/ima2png.py](src/preprocessing/conversion/ima2png.py) — DICOM/IMA → PNG slices + geometry.json
- [src/preprocessing/conversion/png2ima.py](src/preprocessing/conversion/png2ima.py) — PNG slices + geometry.json → .nii.gz
- [src/preprocessing/conversion/mask2datumaro.py](src/preprocessing/conversion/mask2datumaro.py) — Segmentation masks → Datumaro format
- [src/preprocessing/segmentation/segmentation.py](src/preprocessing/segmentation/segmentation.py) — Wood defect segmentation
- [src/preprocessing/segmentation/seg_config.py](src/preprocessing/segmentation/seg_config.py) — Segmentation parameter defaults
- [run](run) — Full automated pipeline script
- [src/ground_truth](src/ground_truth) — Sample input data (IMA files)

## Requirements

- Python 3.10+
- Poetry
- (For training) PyTorch build matching your CPU/CUDA setup

## Install

```bash
poetry install
```

If you need a specific PyTorch build (recommended for GPU training), install it after `poetry install`.

## Quick Start

Run the full automated pipeline on your data:

```bash
./run
```

Or with custom options:

```bash
./run --masks kura,suk --skip-extract
```

## Usage

Convert DICOM/IMA series to PNG (mirrors input folder structure):

```bash
poetry run python src/preprocessing/conversion/ima2png.py \
	--input src/ground_truth \
	--output src/png
```

If the script is executable, you can also run:

```bash
poetry run src/preprocessing/conversion/ima2png.py \
	--input src/ground_truth \
	--output src/png
```

Convert PNG back to NIfTI:

```bash
poetry run python src/preprocessing/conversion/png2ima.py \
	--input src/png \
	--output src/output
```

Generate segmentation masks for wood defects:

```bash
poetry run python src/preprocessing/segmentation/segmentation.py \
	--tree dub1
```

**Selective Mask Generation:**

Generate only specific masks (saves processing time):

```bash
# Generate only bark and background
poetry run python src/preprocessing/segmentation/segmentation.py \
	--tree dub1 \
	--masks kura pozadi

# Generate only cracks
poetry run python src/preprocessing/segmentation/segmentation.py \
	--tree dub1 \
	--masks trhlina
```

Segmentation now uses a fixed pipeline with only two CLI inputs:

- `--tree` selects the source folder under `src/png/<tree>` and writes to `src/output/<tree>`
- `--masks` selects which mask types to export

**Output Masks:**

The segmentation generates 5 mask types:

- `pozadi/` — Background (non-wood areas)
- `kura/` — Bark/crust (outer layer)
- `suk/` — Knots (bright circular regions)
- `hniloba/` — Decay/rot (within knot regions)
- `trhlina/` — Cracks (elongated linear defects detected via gradient analysis)

All masks use binary format: 255 for detected features, 0 for background.

**Convert Masks to Datumaro Format:**

Convert segmentation masks to Datumaro format for use with CVAT or other annotation tools:

```bash
poetry run python src/preprocessing/conversion/mask2datumaro.py \
	--segmentation-output src/output/dub1 \
	--output src/output/datumaro_dub1.zip \
	--task-name dub1
```

The script automatically detects which masks are available and handles partial mask sets.

## Full Pipeline

Use the `run` script to execute the complete pipeline automatically:

```bash
# Run full pipeline (extract → convert → segment → export)
./run

# Generate only specific masks
./run --masks kura,pozadi

# Skip extraction and conversion if files already exist
./run --skip-extract --skip-convert

# Enable CVAT upload (requires CVAT_TOKEN and CVAT_PROJECT_ID in .env)
./run --upload

# Combined example
./run --masks trhlina,suk --skip-extract --upload

# Useful
./run --skip-extract --skip-convert --upload --masks pozadi  

# Import only annotations to an existing CVAT job
./run --skip-extract --skip-convert --upload --upload-job-id 12345 --masks pozadi

# Import only one tree annotations to one existing CVAT job
./run --tree dub5 --skip-extract --skip-convert --upload --upload-job-id 12345 --masks pozadi
```

**Pipeline Options:**

- `--masks, -m MASKS` — Masks to generate (comma-separated: pozadi,kura,suk,hniloba,trhlina or 'all')
- `--tree, -t TREE` — Process only one tree (for example: dub5)
- `--skip-extract` — Skip extraction if PNG files already exist
- `--skip-convert` — Skip IMA→PNG conversion for existing files
- `--upload` — Upload results to CVAT (requires credentials)
- `--upload-job-id JOB_ID` — Import annotations only into an existing CVAT job
- `--datumaro-no-media` — Export Datumaro without images (annotations only)
- `-h, --help` — Show help message

The pipeline automatically:
1. Extracts IMA files from zip archives
2. Converts IMA to PNG format
3. Runs segmentation with specified masks
4. Exports to Datumaro format
5. Optionally uploads to CVAT

## Configuration

**Environment Variables:**

For CVAT upload support, create a `.env` file in the project root:

```bash
CVAT_TOKEN=your_api_token_here
CVAT_PROJECT_ID=your_project_id
CVAT_JOB_ID=your_existing_job_id
CVAT_ORGANIZATION=BP

# Optional upload tuning (defaults shown)
CVAT_MAX_CHUNK_MB=180
CVAT_UPLOAD_ATTEMPTS=5
CVAT_CONNECT_TIMEOUT=30
CVAT_READ_TIMEOUT=1800
```

Then use `./run --upload` to automatically upload results.

When using `--upload-job-id` (or `CVAT_JOB_ID`), the pipeline exports annotation-only Datumaro by default (no media files inside the zip), which is intended for importing labels into an existing CVAT job.

**Segmentation Configs:**

Segmentation defaults are defined in [src/preprocessing/segmentation/seg_config.py](src/preprocessing/segmentation/seg_config.py).

Example parameter structure:
```ini
[segmentation]
# Log extraction
min_log_area = 127000
log_close_kernel_size = 0

# Bark segmentation
crust_alpha = 1.45
crust_beta = -50
crust_wood_thresh = 220

# Crack detection
crack_threshold = 109
crack_max_aspect_ratio = 0.5
crack_max_roundness = 0.9
```

Optional: convert a single series folder or a single file:

```bash
poetry run python src/preprocessing/conversion/ima2png.py --target src/ground_truth/dub1
```

## Examples

**Example 1: Process new data with default settings**
```bash
# Place your .zip files in src/ground_truth/
./run
```

**Example 2: Reprocess with different masks**
```bash
# Already have PNGs, just regenerate specific masks
./run --skip-extract --skip-convert --masks trhlina,hniloba
```

**Example 3: Quick crack-only analysis**
```bash
# Skip all preprocessing, generate only crack masks
./run --skip-extract --skip-convert 
```

**Example 4: Full pipeline with upload**
```bash
# Process and upload to CVAT
./run --upload
```

**Example 5: Segmentation for one tree with selected masks**
```bash
# Run segmentation for one tree and export only selected masks
poetry run python src/preprocessing/segmentation/segmentation.py \
	--tree dub2 \
	--masks kura trhlina
```

**Example 6: Upload to cvat**
```bash
 poetry run python src/preprocessing/upload_to_cvat.py
```

## Notes

- Output PNGs are stored under task folders (e.g., subset1, subset2) to keep large series manageable

## nnU-Net v2 (3D) Integration

This repository now includes a dedicated nnU-Net pipeline under `src/nn_UNet/` with defaults tuned for `3d_fullres`.

Residual Encoder presets are supported (`M`, `L`, `XL`) and default to `L`.
When using these presets, please cite:

Isensee, F.*, Wald, T.*, Ulrich, C.*, Baumgartner, M.*, Roy, S., Maier-Hein, K., Jaeger, P. (2024).
nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation.
arXiv:2404.09556.

### Layout

- `src/preprocessing/conversion/segmentmask2nnunetformat.py` - converts `datasets/Dataset001/dub*` slices into nnU-Net raw NIfTI volumes
- `src/preprocessing/conversion/nnunet_predict.py` - converts one `src/png/dub*` tree to NIfTI inputs, restores predicted masks, and supports Datumaro/CVAT export
- `src/nn_UNet/pipeline.py` - wrappers for prepare, plan/preprocess, train, and predict
- `run_nnunet` - convenience launcher (`poetry run python src/nn_UNet/pipeline.py ...`)

Prepared nnU-Net data will be stored in:

- `src/nn_UNet/nnunet_data/nnUNet_raw/`
- `src/nn_UNet/nnunet_data/nnUNet_preprocessed/`
- `src/nn_UNet/nnunet_data/nnUNet_results/`

### Train on Dataset001 (3D)

Prepare dataset from `Dataset001`:

```bash
./run_nnunet prepare --overwrite
```

Plan and preprocess:

```bash
./run_nnunet plan --verify-dataset-integrity
```

Select a specific Residual Encoder preset:

```bash
./run_nnunet plan --resenc-preset M
./run_nnunet plan --resenc-preset L
./run_nnunet plan --resenc-preset XL
```

If preprocessing runs out of RAM, limit configs and worker processes:

```bash
./run_nnunet plan --configurations 3d_lowres --num-processes 1
```

Train fold 0 using 3D full resolution (default configuration):

```bash
./run_nnunet train --fold 0
```

Resume training from checkpoint (continue from latest checkpoint_latest.pth):

```bash
# Resume 2D training
./run_nnunet train --configuration 2d --fold 0 --plans-identifier nnUNetResEncUNetLPlans --continue-training
./run_nnunet train --configuration 2d --fold 0 --plans-identifier nnUNetResEncUNetLPlans --continue-training --save-every 10 --initial-lr 0.003
# Resume 3D training
./run_nnunet train --fold 0 --continue-training
```

Predict a whole tree, write segmentation-style masks to `src/output/<tree>`, and create a Datumaro zip. If `src/png/<tree>` is missing, the command first converts the matching source from `src/ground_truth` using the preprocessing conversion pipeline:

```bash
./run_nnunet predict-tree --tree dub5 --configuration 2d --fold 0 --plans-identifier nnUNetResEncUNetLPlans --make-datumaro
```

Prediction now uses an automatic fast profile by default:

- `--disable_tta` is applied automatically for speed
- worker counts (`-npp`, `-nps`) are selected from detected GPU VRAM
- no extra tuning flags are required for the common workflow

You can also point it at a different ground-truth root if needed:

```bash
./run_nnunet predict-tree --tree dub5 --ground-truth-root path/to/ground_truth --configuration 2d --fold 0 --plans-identifier nnUNetResEncUNetLPlans --make-datumaro
```

Upload that Datumaro archive to CVAT using the existing `.env` settings:

```bash
./run_nnunet predict-tree --tree dub5 --configuration 2d --fold 0 --plans-identifier nnUNetResEncUNetLPlans --upload-cvat
```

Or run all three steps in one command:

```bash
./run_nnunet all --overwrite --fold 0
```

Low-memory variant (preprocess only selected config with a single worker):

```bash
./run_nnunet all --overwrite --fold 0 --configuration 3d_lowres --plan-configurations 3d_lowres --plan-num-processes 1
```

### Training Profiles (At PC vs Away)

If data is already prepared, skip that step:

```bash
./run_nnunet all --skip-prepare --overwrite --fold 0 --configuration 3d_lowres --plan-configurations 3d_lowres --plan-num-processes 1
```

Use this when you are actively using the computer (lower CPU pressure, shorter run):

```bash
nnUNet_n_proc_DA=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
./run_nnunet train --resenc-preset L --configuration 3d_lowres --fold 0 --trainer nnUNetTrainer_50epochs
```

Use this when you are away and want a full run (higher load, better final quality), or to resume an existing 2D training run:

```bash
nnUNet_n_proc_DA=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 ./run_nnunet train --configuration 2d --fold 0 --plans-identifier nnUNetResEncUNetLPlans --continue-training
```

By default, `run_nnunet train` prefers the plans matching your preset (`nnUNetResEncUNetM/L/XLPlans`).
If only legacy `nnUNetPlans` are available, the wrapper logs a warning and uses those plans.

### Predict

Inference expects nnU-Net style input files named like `case_0000.nii.gz` in an input folder.

```bash
./run_nnunet predict --input path/to/imagesTs --output path/to/predictions --fold 0
```

### Notes about Dataset001 conversion

- One 3D case is generated per folder (`dub1`, `dub5`, `dub11`, ...)
- Source images are read from `SegmentationObject/*.png`
- Labels are read from `SegmentationClass/*.png` and converted from RGB colors via `labelmap.txt`
- Voxel spacing is loaded from `src/png/<series>/geometry.json` when available
- Geometry metadata is saved as geometry.json at the series root
- Segmentation supports selective mask generation to optimize processing time
- Crack detection uses gradient-based analysis with geometric descriptor filtering
- Segmentation parameters are defined in `src/preprocessing/segmentation/seg_config.py`
- The pipeline intelligently skips steps when output already exists
- Datumaro export automatically handles partial mask sets

## Segmentation Features

**Mask Types:**
1. **pozadi** (Background) — Non-wood areas, including the central hole
2. **kura** (Bark) — Outer crust layer detected via contrast enhancement
3. **suk** (Knots) — Bright circular regions in inner wood
4. **hniloba** (Decay) — Dark rot areas within knots
5. **trhlina** (Cracks) — Linear defects detected via Sobel gradient analysis

**Key Algorithms:**
- Log extraction using Otsu thresholding
- Bark segmentation via contrast enhancement and morphological operations
- Knot detection using intensity thresholding with Gaussian smoothing
- Crack detection using gradient magnitude analysis with shape filtering
- Decay detection using two-threshold segmentation within knot regions

**Configurable Parameters:**

Each mask type has tunable parameters in config files:
- Threshold values for intensity-based segmentation
- Kernel sizes for morphological operations
- Minimum area filters for noise removal
- Aspect ratio and roundness thresholds for shape classification
