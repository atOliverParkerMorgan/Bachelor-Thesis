#!/usr/bin/env python3
import argparse
import tempfile
import shutil
from pathlib import Path
import sys
# C:\Users\olive\vscode\bp-main\src\nn_UNet\predictions\DUB_4.nii.gz
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
print(f"Project root added to path: {PROJECT_ROOT}")
from src.nn_UNet.nnunet_predict import export_prediction_masks, export_datumaro_for_tree

def main():
    parser = argparse.ArgumentParser(description="Convert an existing 3D nnU-Net prediction NIfTI to Datumaro format.")
    parser.add_argument("--prediction-nifti", type=Path, required=True, help="Path to the predicted .nii.gz file")
    parser.add_argument("--png-tree-dir", type=Path, required=True, help="Path to the folder containing original slice_*.png files")
    parser.add_argument("--dataset-json", type=Path, required=True, help="Path to the nnU-Net dataset.json")
    parser.add_argument("--output-zip", type=Path, required=True, help="Path where the output datumaro zip will be saved")
    parser.add_argument("--tree-name", type=str, default="tree", help="Name of the tree/task (e.g., DUB_4)")
    args = parser.parse_args()

    if not args.prediction_nifti.exists():
        raise FileNotFoundError(f"Prediction not found: {args.prediction_nifti}")
    if not args.png_tree_dir.exists():
        raise FileNotFoundError(f"PNG Tree dir not found: {args.png_tree_dir}")

    # Create a temporary structure because your export functions expect specific naming conventions
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        temp_pred_dir = temp_path / "preds"
        temp_pred_dir.mkdir()
        temp_seg_out = temp_path / "seg_out"

        # Copy the prediction to the temp folder and rename it to what export_prediction_masks expects
        target_pred_file = temp_pred_dir / f"{args.tree_name}.nii.gz"
        shutil.copy2(args.prediction_nifti, target_pred_file)

        print(f"Extracting 2D PNG masks from {args.prediction_nifti.name}...")
        export_prediction_masks(
            prediction_dir=temp_pred_dir,
            tree_dir=args.png_tree_dir,
            segmentation_output_dir=temp_seg_out,
            dataset_json_path=args.dataset_json,
            tree_name=args.tree_name,
            is_3d=True  # Assuming you ran 3d_fullres
        )

        print(f"Zipping to Datumaro format at {args.output_zip}...")
        export_datumaro_for_tree(
            segmentation_output_dir=temp_seg_out,
            datumaro_zip=args.output_zip,
            tree_name=args.tree_name
        )
        print("Done!")

if __name__ == "__main__":
    main()