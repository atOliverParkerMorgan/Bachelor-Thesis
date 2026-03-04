#!/usr/bin/env python3
import argparse
import sys
import json
import numpy as np
import pydicom
from PIL import Image
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# --- Configuration ---
DEFAULT_INPUT = "src/ground_truth"
DEFAULT_OUTPUT = "src/png"
DEFAULT_LAYOUT = "series"


def hu_from_dicom(ds):
    """Convert raw pixel data to Hounsfield Units with slope/intercept."""
    try:
        img = ds.pixel_array.astype(np.float32)
        slope = float(ds.get("RescaleSlope", 1.0))
        intercept = float(ds.get("RescaleIntercept", 0.0))
        return img * slope + intercept
    except Exception:
        # Fallback for files without pixel data or slope
        # print(f"\tWarning: Could not convert HU ({e})") # Reduced verbosity
        return None


def apply_auto_contrast(hu):
    """
    Robust auto-windowing using percentiles to ignore outliers.
    """
    if hu is None:
        return None

    min_v = np.percentile(hu, 1)
    max_v = np.percentile(hu, 99)

    if max_v == min_v:
        return np.zeros(hu.shape, dtype=np.uint8)

    hu = np.clip(hu, min_v, max_v)
    return ((hu - min_v) / (max_v - min_v) * 255).astype(np.uint8)


def series_id_from_rel_path(rel_path):
    return "__".join(rel_path.parts)


def slice_filename(series_id, idx, flat_layout):
    if flat_layout:
        return f"{series_id}__slice_{idx:04d}.png"
    return f"slice_{idx:04d}.png"


def geometry_output_path(output_root, series_id, flat_layout, series_output_dir):
    if flat_layout:
        geometry_dir = output_root / "geometry"
        geometry_dir.mkdir(parents=True, exist_ok=True)
        return geometry_dir / f"{series_id}.json"
    return series_output_dir / "geometry.json"


def process_series(series_path, dicom_files, output_root, input_root, layout):
    """
    Process a single folder (series) of DICOM files.
    """
    # Resolve paths to absolute to ensure relative_to works
    input_root = input_root.resolve()
    output_root = output_root.resolve()

    # Calculate output path by mirroring the input structure
    try:
        rel_path = series_path.relative_to(input_root)
    except ValueError:
        # If target path is outside input_root, use the folder name
        rel_path = Path(series_path.name)

    flat_layout = layout == "flat"
    series_id = series_id_from_rel_path(rel_path)

    current_output_dir = output_root if flat_layout else output_root / rel_path

    # We create the parent output dir, flat layout writes directly to output_root
    current_output_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(f"Output to: {current_output_dir}")

    # 1. Read headers to sort
    slices = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)

            # Must have ImagePositionPatient for 3D sorting
            if "ImagePositionPatient" not in ds:
                # print(f"\tSkipping {f.name}: No ImagePositionPatient")
                continue

            slices.append(
                {
                    "path": f,
                    "z_pos": float(ds.ImagePositionPatient[2]),
                    "instance": ds.get("InstanceNumber", 0),
                }
            )
        except Exception as e:
            tqdm.write(f"\tSkipping corrupt file {f.name}: {e}")

    # 2. Sort by Z position
    slices.sort(key=lambda s: s["z_pos"])

    if not slices:
        tqdm.write("\tNo valid image slices found in this folder.")
        return

    # 3. Calculate Geometry
    try:
        first_ds = pydicom.dcmread(slices[0]["path"])

        # Calculate Z-spacing
        calc_z_spacing = 0.0
        if len(slices) > 1:
            calc_z_spacing = abs(slices[1]["z_pos"] - slices[0]["z_pos"])

        if calc_z_spacing == 0:
            calc_z_spacing = float(first_ds.get("SliceThickness", 1.0))

        # Handle missing PixelSpacing
        pixel_spacing = first_ds.get("PixelSpacing", [1.0, 1.0])

        geometry = {
            "spacing": [
                float(pixel_spacing[0]),
                float(pixel_spacing[1]),
                round(calc_z_spacing, 5),
            ],
            "origin": list(map(float, first_ds.get("ImagePositionPatient", [0, 0, 0]))),
            "direction": list(
                map(float, first_ds.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0]))
            ),
            "dimensions": [first_ds.Rows, first_ds.Columns, len(slices)],
            "num_slices": len(slices),
            "original_folder": str(series_path),
        }

        # Save geometry alongside the series, or in output_root/geometry for flat layout
        geometry_path = geometry_output_path(
            output_root, series_id, flat_layout, current_output_dir
        )
        with open(geometry_path, "w") as f:
            json.dump(geometry, f, indent=2)

    except Exception as e:
        tqdm.write(f"\tError calculating geometry: {e}")

    # 4. Convert Images
    tqdm.write(f"\tConverting {len(slices)} slices...")

    for idx, slice_info in enumerate(
        tqdm(slices, desc="Converting slices", unit="slice"), start=1
    ):
        try:
            ds = pydicom.dcmread(slice_info["path"])
            hu = hu_from_dicom(ds)
            img8 = apply_auto_contrast(hu)

            if img8 is not None:
                filename = slice_filename(series_id, idx, flat_layout)
                Image.fromarray(img8).save(current_output_dir / filename)

        except Exception as e:
            tqdm.write(f"\tError converting slice {idx}: {e}")
            continue

    tqdm.write("\tDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to PNG with structure mirroring."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help=f"Root input directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Root output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=Path,
        help="Specific subfolder or file to convert (optional)",
    )
    parser.add_argument(
        "--layout",
        choices=["series", "flat"],
        default=DEFAULT_LAYOUT,
        help="Output layout: series (default) mirrors input; flat writes all PNGs into the output root.",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input directory '{args.input}' does not exist.")
        sys.exit(1)

    # Determine files to process
    search_path = args.target if args.target else args.input

    if search_path.is_file():
        files = [search_path]
    else:
        # Recursively find all potential DICOM files
        # We search specifically for likely dicom patterns to speed up initial glob
        extensions = ["*.IMA", "*.dcm", "*.dicom"]
        files = []
        for ext in extensions:
            files.extend(list(search_path.rglob(ext)))

        # Also check files without extensions if the list is empty or comprehensive scan requested
        if not files:
            files.extend(
                [f for f in search_path.rglob("*") if f.is_file() and not f.suffix]
            )

    # Filter for valid files (ignore folders and hidden files)
    files = [f for f in files if f.is_file() and not f.name.startswith(".")]

    if not files:
        print("No files found.")
        sys.exit(0)

    # Group files by their parent directory (series)
    series_map = defaultdict(list)
    for f in files:
        series_map[f.parent].append(f)

    print(f"Found {len(series_map)} series to convert.")

    for series_path, dicom_files in series_map.items():
        process_series(
            series_path=series_path,
            dicom_files=dicom_files,
            output_root=args.output,
            input_root=args.input,
            layout=args.layout,
        )


if __name__ == "__main__":
    main()
