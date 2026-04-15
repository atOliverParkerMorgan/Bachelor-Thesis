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
DEFAULT_INPUT = "src/nn_UNet/predictions/"
DEFAULT_OUTPUT = "src/nn_UNet/predictions/mask"
DEFAULT_TREE = "dub_4"

# nnU-Net class index mapping used in this repository:
# 0 Zdravé dřevo, 1 Pozadí, 2 suk, 3 Hniloba, 4 Kůra, 5 Trhlina, 6 Poškození hmyzem
LABEL_TO_RGB = {
    0: (163, 56, 212),   # Zdravé dřevo   #a338d4
    1: (214, 149, 170),  # Pozadí         #d695aa
    2: (174, 60, 29),    # Suk            #ae3c1d
    3: (153, 242, 107),  # Hniloba        #99f26b
    4: (38, 114, 129),   # Kůra           #267281
    5: (222, 137, 84),   # Trhlina        #de8954
    6: (63, 19, 205),    # Poškození hmyzem #3f13cd
}


def _strip_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    if name.lower().endswith(".nii"):
        return name[:-4]
    return path.stem


def _find_nifti_files(input_path: Path) -> list[Path]:
    nii_gz = list(input_path.rglob("*.nii.gz"))
    nii = list(input_path.rglob("*.nii"))
    files = [f for f in nii_gz + nii if f.is_file() and not f.name.startswith(".")]
    return sorted(files, key=lambda p: str(p).lower())


def _clean_output_pngs(output_dir: Path) -> None:
    # Remove stale slices so output always matches the current volume depth.
    for png_file in output_dir.glob("*.png"):
        try:
            png_file.unlink()
        except Exception as e:
            tqdm.write(f"\tWarning: could not remove stale file {png_file.name}: {e}")


def _labels_to_color_mask(slice_2d: np.ndarray) -> np.ndarray:
    labels = np.rint(slice_2d).astype(np.int16)
    colored = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

    for label_id, rgb in LABEL_TO_RGB.items():
        colored[labels == label_id] = rgb

    return colored


def _candidate_tree_names(tree_name: str) -> list[str]:
    candidates = [tree_name, tree_name.lower(), tree_name.replace("_", ""), tree_name.replace("_", "").lower()]
    unique_candidates: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _find_reference_images_dir(tree_name: str, output_root: Path) -> Path | None:
    search_roots = [
        Path("output"),
        Path("src") / "output",
        output_root.parent,
        output_root.parent.parent / "output",
    ]

    for base_dir in search_roots:
        for candidate_tree_name in _candidate_tree_names(tree_name):
            candidate_dir = base_dir / candidate_tree_name / "images"
            if candidate_dir.is_dir():
                return candidate_dir

    return None

def process_nifti(nifti_path: Path, output_root: Path, tree_name: str):
    """
    Reads a 3D NIfTI file and exports each Z-slice as a PNG.
    """
    # Create a subfolder named after the NIfTI file (e.g., DUB_4_0000)
    series_id = _strip_nifti_suffix(nifti_path)
    current_output_dir = output_root / series_id
    current_output_dir.mkdir(parents=True, exist_ok=True)
    _clean_output_pngs(current_output_dir)

    reference_images_dir = _find_reference_images_dir(tree_name, output_root)
    if reference_images_dir is None:
        tqdm.write(
            f"\tWarning: could not find an images folder for tree '{tree_name}' "
            "under output/{tree}/images. Keeping all slices."
        )
        allowed_slice_names = None
    else:
        allowed_slice_names = {
            png_file.name
            for png_file in reference_images_dir.glob("*.png")
            if png_file.is_file()
        }
        if not allowed_slice_names:
            tqdm.write(
                f"\tWarning: found images folder {reference_images_dir} but it contains no PNG files. "
                "All slices will be removed from the result."
            )
        else:
            tqdm.write(
                f"\tUsing {len(allowed_slice_names)} reference slice(s) from {reference_images_dir}."
            )

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

    known_values = set(LABEL_TO_RGB.keys())
    unique_values = set(np.unique(np.rint(volume).astype(np.int16)).tolist())
    unknown_values = sorted(unique_values - known_values)
    if unknown_values:
        tqdm.write(
            f"\tWarning: {nifti_path.name} contains unknown label(s): {unknown_values}. "
            "Unknown labels are written as black."
        )

    tqdm.write(f"\tExtracting {num_slices} slices...")

    for z in tqdm(range(num_slices), desc="Slicing NIfTI", unit="slice"):
        try:
            # Extract the 2D slice
            slice_2d = volume[z, :, :]

            filename = f"slice_{z + 1:04d}.png"
            if allowed_slice_names is not None and filename not in allowed_slice_names:
                continue

            # Convert label-index slice into a color segmentation mask.
            slice_img = _labels_to_color_mask(slice_2d)

            Image.fromarray(slice_img, mode="RGB").save(current_output_dir / filename)
            
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
    parser.add_argument(
        "--tree",
        "-s",
        type=str,
        default=DEFAULT_TREE,
        help=(
            "Tree stem to process when --input is a directory "
            f"(case-insensitive, default: {DEFAULT_TREE})"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all NIfTI files under --input directory (ignore --series).",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: Input path '{args.input}' does not exist.")
        sys.exit(1)

    # Determine files to process
    if args.input.is_file():
        files = [args.input]
    else:
        all_files = _find_nifti_files(args.input)
        if args.all:
            files = all_files
        else:
            target_tree = args.tree.lower()
            files = [
                f for f in all_files
                if _strip_nifti_suffix(f).lower() == target_tree
            ]

    if not files:
        if args.input.is_file():
            print(f"No valid NIfTI file found at {args.input}")
        elif args.all:
            print(f"No NIfTI files found in {args.input}")
        else:
            print(
                "No matching NIfTI file found for "
                f"tree '{args.tree}' in {args.input} (case-insensitive)."
            )
        sys.exit(0)

    print(f"Found {len(files)} NIfTI file(s) to convert.")

    for nifti_file in files:
        process_nifti(
            nifti_path=nifti_file,
            output_root=args.output,
            tree_name=args.tree,
        )

if __name__ == "__main__":
    main()