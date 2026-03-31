#!/usr/bin/env python3
"""Run only the NIfTI to Datumaro conversion for an existing prediction."""

import shutil
import tempfile
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

def main():
    tree_name = "DUB_4"
    
    # Define your paths based on your previous terminal logs
    ground_truth_root = Path("src/ground_truth")
    prediction_dir = Path("src/nn_UNet/predictions")
    dataset_json_path = Path("src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/dataset.json")
    
    # We will output the fixed files to a new folder so we don't mix them up
    segmentation_output_root = Path("src/nn_UNet/predictions_fixed")

    print(f"Starting conversion recovery for {tree_name}...")
    
    if not (prediction_dir / f"{tree_name}.nii.gz").exists():
        print(f"Error: Could not find {tree_name}.nii.gz in {prediction_dir}")
        return

    # Create a temporary directory to rebuild the PNGs
    temp_dir = Path(tempfile.mkdtemp(prefix=f"nnunet_repack_{tree_name}_"))
    png_root = temp_dir / "pngs"

    try:
        print("1. Re-extracting PNGs from ground truth zip (Skipping GPU Prediction)...")
        tree_dir = prepare_png_tree_from_ground_truth(
            tree_name=tree_name,
            png_root=png_root,
            ground_truth_root=ground_truth_root,
            temp_root=temp_dir
        )

        print("2. Slicing NIfTI and generating correct case-insensitive folders...")
        export_prediction_masks(
            prediction_dir=prediction_dir,
            tree_dir=tree_dir,
            segmentation_output_dir=segmentation_output_root,
            dataset_json_path=dataset_json_path,
            tree_name=tree_name,
            is_3d=True
        )

        print("3. Zipping to Datumaro format...")
        datumaro_zip = default_datumaro_output(segmentation_output_root, tree_name)
        export_datumaro_for_tree(segmentation_output_root, datumaro_zip, tree_name)
        
        print(f"\nSuccess! Fixed Datumaro dataset zipped at: {datumaro_zip}")
        print(f"You can view the raw generated folders at: {segmentation_output_root}/masks/")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Cleaned up temporary PNG files.")

if __name__ == "__main__":
    main()