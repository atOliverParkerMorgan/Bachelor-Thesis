#!/usr/bin/env python3
import argparse
import sys
import json
import re
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
DEFAULT_INPUT = "src/png"  # Where your PNGs/Geometry are
DEFAULT_OUTPUT = "src/segmentations"  # Where you want the NIfTI files
DEFAULT_LAYOUT = "series"


def get_3d_direction(dicom_orientation):
    """
    Convert DICOM ImageOrientationPatient (6 floats) to ITK 3D Direction (9 floats).
    DICOM stores [Xx, Xy, Xz, Yx, Yy, Yz].
    ITK needs [Xx, Xy, Xz, Yx, Yy, Yz, Zx, Zy, Zz].
    We compute the Z vector using the cross product of X and Y.
    """
    if len(dicom_orientation) != 6:
        # Fallback to identity if data is missing/corrupt
        return [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    rx = np.array(dicom_orientation[0:3])
    ry = np.array(dicom_orientation[3:6])
    rz = np.cross(rx, ry)  # Compute Z orthogonal to X and Y

    # Normalize just in case
    rx /= np.linalg.norm(rx)
    ry /= np.linalg.norm(ry)
    rz /= np.linalg.norm(rz)

    # SimpleITK expects a flat list of 9 elements
    return np.concatenate([rx, ry, rz]).tolist()


def process_series(series_folder, output_root, input_root, layout):
    """
    Reconstruct a single folder of PNGs into a .nii.gz file.
    """
    if layout == "flat":
        raise ValueError("Flat layout requires grouped series processing.")

    geo_path = series_folder / "geometry.json"

    # 1. Validation
    if not geo_path.exists():
        tqdm.write(f"\tSkipping {series_folder.name}: No geometry.json found.")
        return

    # 2. Determine Output Path
    # Mirror structure: input/scan1 -> output/scan1.nii.gz
    try:
        rel_path = series_folder.relative_to(input_root)
    except ValueError:
        rel_path = Path(series_folder.name)

    # Output file will be named after the folder (e.g., "dub1.nii.gz")
    output_file = output_root / rel_path.with_suffix(".nii.gz")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"\nProcessing Series: {series_folder}")
    tqdm.write(f"Output to: {output_file}")

    # 3. Load Geometry
    try:
        with open(geo_path, "r") as f:
            geo = json.load(f)
    except Exception as e:
        tqdm.write(f"\tError reading geometry: {e}")
        return

    # 4. Load PNGs
    # Find all PNGs and sort them (assuming format slice_0001.png, slice_0002.png, etc.)
    png_files = sorted(list(series_folder.glob("slice_*.png")))

    if not png_files:
        tqdm.write("\tNo 'slice_*.png' files found.")
        return

    tqdm.write(f"\tLoading {len(png_files)} slices...")

    # Read first image to get dimensions/type
    first_img = np.array(Image.open(png_files[0]))

    # Pre-allocate volume (D, H, W)
    volume_shape = (len(png_files), first_img.shape[0], first_img.shape[1])
    volume_data = np.zeros(volume_shape, dtype=first_img.dtype)

    # Load data into volume
    for i, p in enumerate(tqdm(png_files, desc="Loading slices", unit="slice")):
        volume_data[i, :, :] = np.array(Image.open(p))

    # 5. Convert to SimpleITK Image
    itk_img = sitk.GetImageFromArray(volume_data)

    # 6. Apply Metadata
    # Spacing (X, Y, Z)
    if "spacing" in geo:
        itk_img.SetSpacing(geo["spacing"])

    # Origin (X, Y, Z)
    if "origin" in geo:
        itk_img.SetOrigin(geo["origin"])

    # Direction (3x3 Matrix)
    if "direction" in geo:
        direction_3d = get_3d_direction(geo["direction"])
        itk_img.SetDirection(direction_3d)

    # 7. Write to Disk
    try:
        sitk.WriteImage(itk_img, str(output_file))
        tqdm.write("\tSuccess! Saved 3D volume.")
    except Exception as e:
        tqdm.write(f"\tError writing NIfTI: {e}")


def parse_flat_series(png_files):
    series_map = {}
    for png_path in png_files:
        match = re.match(r"^(.*)__slice_(\d+)\.png$", png_path.name)
        if not match:
            continue
        series_id = match.group(1)
        series_map.setdefault(series_id, []).append(png_path)
    return series_map


def load_geometry(flat_input_root, series_id):
    geo_path = flat_input_root / "geometry" / f"{series_id}.json"
    if not geo_path.exists():
        return None
    with open(geo_path, "r") as f:
        return json.load(f)


def process_flat_series(series_id, png_files, output_root, input_root):
    geo = load_geometry(input_root, series_id)
    if geo is None:
        tqdm.write(f"\tSkipping {series_id}: No geometry json found.")
        return

    output_file = output_root / f"{series_id}.nii.gz"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    png_files = sorted(png_files)
    tqdm.write(f"\nProcessing Series: {series_id}")
    tqdm.write(f"Output to: {output_file}")

    first_img = np.array(Image.open(png_files[0]))
    volume_shape = (len(png_files), first_img.shape[0], first_img.shape[1])
    volume_data = np.zeros(volume_shape, dtype=first_img.dtype)

    for i, p in enumerate(tqdm(png_files, desc="Loading slices", unit="slice")):
        volume_data[i, :, :] = np.array(Image.open(p))

    itk_img = sitk.GetImageFromArray(volume_data)

    if "spacing" in geo:
        itk_img.SetSpacing(geo["spacing"])
    if "origin" in geo:
        itk_img.SetOrigin(geo["origin"])
    if "direction" in geo:
        direction_3d = get_3d_direction(geo["direction"])
        itk_img.SetDirection(direction_3d)

    try:
        sitk.WriteImage(itk_img, str(output_file))
        tqdm.write("\tSuccess! Saved 3D volume.")
    except Exception as e:
        tqdm.write(f"\tError writing NIfTI: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PNG series + geometry.json back to NIfTI."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help=f"Root input directory containing folders of PNGs (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Root output directory for NIfTI files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target", "-t", type=Path, help="Specific subfolder to process (optional)"
    )
    parser.add_argument(
        "--layout",
        choices=["series", "flat"],
        default=DEFAULT_LAYOUT,
        help="Input layout: series (default) expects geometry.json in each folder; flat expects PNGs in root and geometry in input/geometry/.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory '{args.input}' does not exist.")
        sys.exit(1)

    if args.layout == "flat":
        if args.target:
            print("Error: --target is not supported with flat layout.")
            sys.exit(1)

        png_files = sorted(list(args.input.glob("*.png")))
        if not png_files:
            print("No PNG files found in flat layout.")
            sys.exit(0)

        series_map = parse_flat_series(png_files)
        if not series_map:
            print(
                "No series found in flat layout. Expected filenames like <series>__slice_0001.png"
            )
            sys.exit(0)

        print(f"Found {len(series_map)} series to convert.")
        for series_id, files in series_map.items():
            process_flat_series(series_id, files, args.output, args.input)
        return

    # Identify folders to process
    folders_to_process = []

    if args.target:
        # If user targets specific folder
        if (args.target / "geometry.json").exists():
            folders_to_process.append(args.target)
        else:
            print(f"Error: Target '{args.target}' does not contain geometry.json")
    else:
        # Recursively find all folders containing geometry.json
        for geo_file in args.input.rglob("geometry.json"):
            folders_to_process.append(geo_file.parent)

    if not folders_to_process:
        print("No folders with 'geometry.json' found.")
        sys.exit(0)

    print(f"Found {len(folders_to_process)} series to convert.")

    for folder in folders_to_process:
        process_series(folder, args.output, args.input, args.layout)


if __name__ == "__main__":
    main()
