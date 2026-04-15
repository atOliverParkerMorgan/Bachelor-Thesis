#!/usr/bin/env python3
"""Unified pipeline: nnU-Net NIfTI prediction → Datumaro zip.

Replaces the two-step manual workflow::

    poetry run python src/preprocessing/conversion/nii2mask.py --tree dub_4
    poetry run python src/preprocessing/conversion/mask2datumaro.py ...

Usage example::

    poetry run python src/preprocessing/conversion/predict2datumaro.py --tree DUB_4

The script reads the raw NIfTI label volume produced by nnU-Net, builds
per-class binary mask PNGs using the dataset.json label map, then exports a
Datumaro zip ready for upload to CVAT.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

DEFAULT_PREDICTIONS_ROOT = Path("src/nn_UNet/predictions")
DEFAULT_DATASET_JSON = Path(
    "src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/dataset.json"
)


def _find_nii(predictions_root: Path, tree_name: str) -> Path:
    """Find the NIfTI prediction file, trying common name variants."""
    for candidate in (
        predictions_root / f"{tree_name}.nii.gz",
        predictions_root / f"{tree_name.lower()}.nii.gz",
        predictions_root / f"{tree_name.upper()}.nii.gz",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"NIfTI prediction not found for tree '{tree_name}' in {predictions_root}.\n"
        f"Expected e.g. {predictions_root / tree_name}.nii.gz"
    )


def _find_images_dir(predictions_root: Path, tree_name: str, images_dir: Path | None) -> Path:
    """Resolve the directory containing slice_*.png CT images."""
    if images_dir is not None:
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        return images_dir

    for candidate in (
        predictions_root / tree_name / "images",
        predictions_root / tree_name.lower() / "images",
        predictions_root / tree_name.upper() / "images",
    ):
        if candidate.is_dir() and any(candidate.glob("slice_*.png")):
            return candidate

    raise FileNotFoundError(
        f"CT slice images not found for tree '{tree_name}'.\n"
        f"Expected slice_*.png files under {predictions_root / tree_name}/images/"
    )


def predict2datumaro(
    tree_name: str,
    predictions_root: Path,
    dataset_json_path: Path,
    output_zip: Path,
    task_name: str,
    images_dir: Path | None = None,
    save_media: bool = True,
    item_id_mode: str = "stem",
) -> Path:
    from src.nn_UNet.tree_inference_helpers import export_prediction_masks
    from src.preprocessing.conversion.mask2datumaro import export_datumaro_dataset

    nii_path = _find_nii(predictions_root, tree_name)
    resolved_images_dir = _find_images_dir(predictions_root, tree_name, images_dir)

    print(f"NIfTI prediction : {nii_path}")
    print(f"CT slice images  : {resolved_images_dir}")
    print(f"dataset.json     : {dataset_json_path}")
    print(f"Output zip       : {output_zip}")

    temp_dir = Path(
        tempfile.mkdtemp(prefix=f"predict2datumaro_{tree_name.lower()}_")
    )
    segmentation_output = temp_dir / "segmentation_style"

    try:
        print("\n[1/2] Building per-class binary masks from NIfTI...")
        export_prediction_masks(
            prediction_dir=predictions_root,
            tree_dir=resolved_images_dir,
            segmentation_output_dir=segmentation_output,
            dataset_json_path=dataset_json_path,
            tree_name=tree_name,
            is_3d=True,
        )

        print("\n[2/2] Exporting Datumaro dataset...")
        export_datumaro_dataset(
            segmentation_output=segmentation_output,
            output=output_zip,
            task_name=task_name,
            save_media=save_media,
            item_id_mode=item_id_mode,
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\nDone! Datumaro zip: {output_zip}")
    return output_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert nnU-Net NIfTI predictions to a Datumaro zip in one step.\n"
            "Replaces the manual nii2mask → mask2datumaro workflow."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tree", "-t",
        required=True,
        help="Tree name matching the NIfTI file, e.g. DUB_4",
    )
    parser.add_argument(
        "--predictions-root", "-p",
        type=Path,
        default=DEFAULT_PREDICTIONS_ROOT,
        help=f"Root dir containing <tree>.nii.gz and <tree>/images/ (default: {DEFAULT_PREDICTIONS_ROOT})",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override CT slice directory (default: <predictions-root>/<tree>/images)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output .zip path (default: <predictions-root>/<tree_lower>_datumaro.zip)",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
        help=f"Path to nnU-Net dataset.json (default: {DEFAULT_DATASET_JSON})",
    )
    parser.add_argument(
        "--task-name", "-n",
        type=str,
        default=None,
        help="Datumaro subset/task name (default: lower-cased tree name)",
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Omit image media from the Datumaro zip (annotations only)",
    )
    parser.add_argument(
        "--item-id-mode",
        choices=["stem", "name", "relative_stem", "relative_name"],
        default="stem",
        help="How to build Datumaro item IDs for CVAT frame matching (default: stem)",
    )

    args = parser.parse_args()

    tree_name = args.tree
    tree_slug = tree_name.lower().replace(" ", "_")

    output_zip = args.output or (
        args.predictions_root / f"{tree_slug}_datumaro.zip"
    )
    task_name = args.task_name or tree_slug

    if not args.dataset_json.exists():
        print(f"Error: dataset.json not found at {args.dataset_json}", file=sys.stderr)
        sys.exit(1)

    try:
        predict2datumaro(
            tree_name=tree_name,
            predictions_root=args.predictions_root,
            dataset_json_path=args.dataset_json,
            output_zip=output_zip,
            task_name=task_name,
            images_dir=args.images_dir,
            save_media=not args.no_media,
            item_id_mode=args.item_id_mode,
        )
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
