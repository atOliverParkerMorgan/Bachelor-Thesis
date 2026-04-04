#!/usr/bin/env python3
"""Convert 3D NIfTI volumes/masks into 2D PNG slices."""

import argparse
import sys
import numpy as np
import SimpleITK as sitk
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---
DEFAULT_INPUT = "src/nn_UNet/predictions"
DEFAULT_OUTPUT = "src/nn_UNet/predictions_png"

def process_nifti(nifti_path: Path, output_root: Path):
    """
    Reads a 3D NIfTI file and exports each Z-slice as a PNG.
    """
    # Create a subfolder named after the NIfTI file (e.g., DUB_4_0000)
    series_id = nifti_path.name.replace(".nii.gz", "").replace(".nii", "")
    current_output_dir = output_root / series_id
    current_output_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"Processing: {nifti_path.name}")
    tqdm.write(f"Output to: {current_output_dir}")

    try:
        # Load the NIfTI file
        itk_image = sitk.ReadImage(str(nifti_path))
        # Convert to numpy array. SimpleITK returns shape as (Z, Y, X)
        volume = sitk.GetArrayFromImage(itk_image)
    except Exception as e:
        tqdm.write(f"\tError reading {nifti_path.name}: {e}")
        return

    num_slices = volume.shape[0]
    
    if num_slices == 0:
        tqdm.write("\tNo slices found in volume.")
        return

    tqdm.write(f"\tExtracting {num_slices} slices...")

    for z in tqdm(range(num_slices), desc="Slicing NIfTI", unit="slice"):
        try:
            # Extract the 2D slice
            slice_2d = volume[z, :, :]
            
            # Ensure it is in 8-bit unsigned integer format for standard PNGs
            slice_img = slice_2d.astype(np.uint8)
            
            filename = f"slice_{z + 1:04d}.png"
            Image.fromarray(slice_img, mode="L").save(current_output_dir / filename)
            
        except Exception as e:
            tqdm.write(f"\tError saving slice {z + 1}: {e}")
            continue

    tqdm.write("\tDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D NIfTI files to 2D PNG slices."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help=f"Input directory or specific .nii.gz file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Root output directory (default: {DEFAULT_OUTPUT})",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)

    # Determine files to process
    if args.input.is_file():
        files = [args.input]
    else:
        # Search for NIfTI files
        files = list(args.input.rglob("*.nii.gz")) + list(args.input.rglob("*.nii"))

    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    if not files:
        print(f"No NIfTI files found in {args.input}")
        sys.exit(0)

    print(f"Found {len(files)} NIfTI file(s) to convert.")

    for nifti_file in files:
        process_nifti(
            nifti_path=nifti_file,
            output_root=args.output,
        )

if __name__ == "__main__":
    main()