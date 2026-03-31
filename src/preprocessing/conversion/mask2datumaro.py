#!/usr/bin/env python3
"""Run only the NIfTI to Datumaro conversion for an existing prediction."""

import argparse
import logging
import shutil
import tempfile
from tqdm import tqdm
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3] 
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
# Import your newly fixed helper functions
from src.nn_UNet.tree_inference_helpers import (
    prepare_png_tree_from_ground_truth,
    export_prediction_masks,
    export_datumaro_for_tree,
    default_datumaro_output
)

def repack_tree(
    tree_name: str,
    ground_truth_root: Path,
    prediction_dir: Path,
    dataset_json_path: Path,
    output_root: Path,
) -> Path | None:
    """Core logic to repack a NIfTI prediction into a Datumaro dataset."""
    logger = logging.getLogger(__name__)

    prediction_nifti = prediction_dir / f"{tree_name}.nii.gz"
    if not prediction_nifti.exists():
        logger.error(f"Could not find predicted NIfTI file: {prediction_nifti}")
        return None

    # Create a temporary directory to rebuild the PNGs
    temp_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_repack_{tree_name}_"))
    png_root = temp_dir / "pngs"

    try:
        logger.info(f"Re-extracting PNGs from ground truth zip for {tree_name}...")
        tree_dir = prepare_png_tree_from_ground_truth(
            tree_name=tree_name,
            png_root=png_root,
            ground_truth_root=ground_truth_root,
            temp_root=temp_dir
        )

        logger.info(f"Slicing NIfTI and mapping labels to {output_root}...")
        # export_prediction_masks will handle the heavy lifting (and the fixed label map)
        export_prediction_masks(
            prediction_dir=prediction_dir,
            tree_dir=tree_dir,
            segmentation_output_dir=output_root,
            dataset_json_path=dataset_json_path,
            tree_name=tree_name,
            is_3d=True
        )

        logger.info("Zipping raw masks into Datumaro format...")
        datumaro_zip = default_datumaro_output(output_root, tree_name)
        export_datumaro_for_tree(output_root, datumaro_zip, tree_name)
        
        return datumaro_zip

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug(f"Cleaned up temporary staging directory: {temp_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Repack an existing nnU-Net NIfTI prediction into a Datumaro dataset."
    )
    parser.add_argument(
        "--tree",
        "-t",
        type=str,
        default="DUB_4",
        help="Name of the tree to process (e.g., 'DUB_4').",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=Path,
        default=Path("src/ground_truth"),
        help="Path to the ground truth directory containing the raw zip files.",
    )
    parser.add_argument(
        "--prediction-dir",
        "-p",
        type=Path,
        default=Path("src/nn_UNet/predictions"),
        help="Path where the existing predicted .nii.gz file is located.",
    )
    parser.add_argument(
        "--dataset-json",
        "-j",
        type=Path,
        default=Path("src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/dataset.json"),
        help="Path to the nnU-Net dataset.json for label mapping.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("src/nn_UNet/predictions_fixed"),
        help="Path to save the generated PNG masks and images.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Setup styling for the logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting Datumaro repack process for {args.tree}")
    
    datumaro_zip = repack_tree(
        tree_name=args.tree,
        ground_truth_root=args.ground_truth,
        prediction_dir=args.prediction_dir,
        dataset_json_path=args.dataset_json,
        output_root=args.output_dir,
    )

    if datumaro_zip and datumaro_zip.exists():
        logger.info(f"Success! Datumaro dataset zipped at: {datumaro_zip}")
        logger.info(f"Raw image/mask folders are located at: {args.output_dir}")
    else:
        logger.error("Failed to create the Datumaro dataset.")


if __name__ == "__main__":
    main()