#!/usr/bin/env python3
"""
Split large NIfTI volumes into sub-volumes for nnU-Net training.

This script splits each image/label pair along the longest spatial axis
into N chunks (default 5), with configurable overlap to avoid boundary
artifacts. Output follows nnU-Net naming conventions.

Usage:
    python split_nnunet_volumes.py \
        --input_dir /path/to/nnUNet_raw/Dataset001_BPWoodDefects \
        --output_dir /path/to/nnUNet_raw/Dataset002_BPWoodDefectsSplit \
        --n_splits 5 \
        --overlap_slices 10

Directory structure expected:
    input_dir/
        imagesTr/   (dub_1_0000.nii.gz, dub_11_0000.nii.gz, ...)
        labelsTr/   (dub_1.nii.gz, dub_11.nii.gz, ...)
        dataset.json
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def find_split_axis(img_shape):
    """Return the axis with the most slices (the one to split along)."""
    return int(np.argmax(img_shape[:3]))


def normalize_nnunet_labels(ds_json):
    """Ensure nnU-Net v2 required background label key is present as 'background': 0."""
    labels = ds_json.get("labels")
    if not isinstance(labels, dict):
        return ds_json

    if "background" in labels and labels["background"] == 0:
        return ds_json

    bg_key = None
    for key, value in labels.items():
        if value == 0:
            bg_key = key
            break

    if bg_key is None:
        labels = {"background": 0, **labels}
    else:
        value_0_rest = {k: v for k, v in labels.items() if k != bg_key}
        labels = {"background": 0, **value_0_rest}

    ds_json["labels"] = labels
    return ds_json


def load_json_with_fallback(path):
    """Load JSON with UTF-8 first, then legacy Windows encodings if needed."""
    for encoding in ("utf-8", "cp1250", "latin-1"):
        try:
            with open(path, encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode JSON file: {path}")


def get_split_ranges(total, n_splits, overlap):
    """
    Compute split boundaries for a 1D axis length.

    Returns list of (start_idx, end_idx).
    """
    base_size = total // n_splits
    remainder = total % n_splits

    ranges = []
    pos = 0
    for i in range(n_splits):
        # Distribute remainder slices to first chunks
        chunk_size = base_size + (1 if i < remainder else 0)
        start = max(0, pos - overlap)
        end = min(total, pos + chunk_size + overlap)
        ranges.append((start, end))
        pos += chunk_size

    return ranges


def update_affine_for_subvolume(affine, axis, start_idx):
    """
    Adjust the affine origin so the sub-volume has correct spatial coordinates.
    """
    new_affine = affine.copy()
    # Shift origin by start_idx voxels along the split axis
    new_affine[:3, 3] += start_idx * affine[:3, axis]
    return new_affine


def split_case(img_path, label_path, case_id, split_axis, n_splits, overlap):
    """
    Split one image/label pair. Returns list of
    (sub_img_nifti, sub_label_nifti, new_case_id).
    """
    img_nii = nib.load(str(img_path))
    label_nii = nib.load(str(label_path))

    # Keep source arrays in original dtype and avoid float64 upcast from get_fdata().
    img_data = np.asarray(img_nii.dataobj)
    label_data = np.asarray(label_nii.dataobj)

    affine = img_nii.affine
    header = img_nii.header

    # Determine axis
    if split_axis == "auto":
        axis = find_split_axis(img_data.shape)
    else:
        axis = int(split_axis)

    print(f"  Volume shape: {img_data.shape}, splitting along axis {axis} "
          f"({img_data.shape[axis]} slices) into {n_splits} chunks, "
          f"overlap={overlap}")

    split_ranges = get_split_ranges(img_data.shape[axis], n_splits, overlap)

    results = []
    for i, (start, end) in enumerate(split_ranges):
        img_slicing = [slice(None)] * img_data.ndim
        img_slicing[axis] = slice(start, end)
        img_sub = np.asarray(img_data[tuple(img_slicing)])

        lbl_slicing = [slice(None)] * label_data.ndim
        lbl_slicing[axis] = slice(start, end)
        lbl_sub = np.asarray(label_data[tuple(lbl_slicing)], dtype=np.int16)

        new_affine = update_affine_for_subvolume(affine, axis, start)

        sub_img_nii = nib.Nifti1Image(img_sub, new_affine, header)
        sub_lbl_nii = nib.Nifti1Image(lbl_sub, new_affine, header)

        # New case ID: originalID_partN  (e.g., dub_1_part0)
        new_case_id = f"{case_id}_part{i}"

        print(f"    Part {i}: slices [{start}:{end}] -> shape {img_sub.shape}, "
              f"case_id={new_case_id}")

        results.append((sub_img_nii, sub_lbl_nii, new_case_id))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Split NIfTI volumes for nnU-Net training"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to nnU-Net raw dataset (contains imagesTr/, labelsTr/, dataset.json)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Path for new split dataset"
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Number of sub-volumes per case (default: 5)"
    )
    parser.add_argument(
        "--overlap_slices", type=int, default=10,
        help="Number of overlap slices on each side of a split boundary (default: 10)"
    )
    parser.add_argument(
        "--split_axis", default="auto",
        help="Axis to split along: 0, 1, 2, or 'auto' for longest (default: auto)"
    )
    parser.add_argument(
        "--channel_suffix", default="_0000",
        help="Channel suffix for image files (default: _0000)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    images_in = input_dir / "imagesTr"
    labels_in = input_dir / "labelsTr"

    if not images_in.exists() or not labels_in.exists():
        print(f"ERROR: Expected imagesTr/ and labelsTr/ in {input_dir}")
        sys.exit(1)

    # Create output structure
    images_out = output_dir / "imagesTr"
    labels_out = output_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Discover cases from labelsTr (labels don't have channel suffix)
    label_files = sorted(labels_in.glob("*.nii.gz"))
    print(f"Found {len(label_files)} label files in {labels_in}")

    total_new_cases = 0

    for label_path in label_files:
        # Extract case ID: e.g., dub_1.nii.gz -> dub_1
        case_id = label_path.name.replace(".nii.gz", "")

        # Find corresponding image (with channel suffix)
        img_name = f"{case_id}{args.channel_suffix}.nii.gz"
        img_path = images_in / img_name

        if not img_path.exists():
            print(f"WARNING: No image found for {case_id} (expected {img_path}), skipping")
            continue

        print(f"\nProcessing case: {case_id}")
        print(f"  Image: {img_path.name}")
        print(f"  Label: {label_path.name}")

        results = split_case(
            img_path, label_path, case_id,
            args.split_axis, args.n_splits, args.overlap_slices
        )

        for sub_img, sub_lbl, new_case_id in results:
            # Save image: new_case_id + channel_suffix + .nii.gz
            out_img_path = images_out / f"{new_case_id}{args.channel_suffix}.nii.gz"
            out_lbl_path = labels_out / f"{new_case_id}.nii.gz"

            nib.save(sub_img, str(out_img_path))
            nib.save(sub_lbl, str(out_lbl_path))
            total_new_cases += 1

    # Create new dataset.json
    dataset_json_path = input_dir / "dataset.json"
    if dataset_json_path.exists():
        ds_json = load_json_with_fallback(dataset_json_path)
    else:
        ds_json = {}

    ds_json["numTraining"] = total_new_cases
    ds_json = normalize_nnunet_labels(ds_json)

    # Add a note about the split
    ds_json["_split_info"] = {
        "original_dataset": str(input_dir),
        "n_splits": args.n_splits,
        "overlap_slices": args.overlap_slices,
        "split_axis": args.split_axis,
        "original_num_cases": len(label_files),
        "new_num_cases": total_new_cases,
    }

    out_json_path = output_dir / "dataset.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(ds_json, f, indent=4, ensure_ascii=True)

    print(f"\n{'='*60}")
    print(f"Done! Split {len(label_files)} volumes into {total_new_cases} sub-volumes")
    print(f"Output: {output_dir}")
    print(f"dataset.json updated with numTraining={total_new_cases}")
    print(f"\nNext steps:")
    print(f"  1. Verify the output: check a few sub-volumes in ITK-SNAP or 3D Slicer")
    print(f"  2. Run nnU-Net preprocessing:")
    print(f"     nnUNetv2_plan_and_preprocess -d DATASET_ID -c 3d_fullres")
    print(f"  3. Train:")
    print(f"     nnUNetv2_train DATASET_ID 3d_fullres FOLD")


if __name__ == "__main__":
    main()