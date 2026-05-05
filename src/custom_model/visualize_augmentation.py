#!/usr/bin/env python3
"""Generate augmented NIfTI patches for inspection in 3D Slicer.

Usage
-----
  poetry run python src/custom_model/visualize_augmentation.py \
      path/to/image_0000.nii.gz \
      path/to/label.nii.gz \
      --output-dir augmented_previews \
      --n-samples 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.custom_model.transforms import get_train_transforms


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return np.asarray(t)


def _save_nifti(array: np.ndarray, spacing: tuple, out_path: Path, is_label: bool) -> None:
    if array.ndim == 4:
        array = array[0]
    arr = array.astype(np.uint8 if is_label else np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(out_path))
    print(f"  {'label' if is_label else 'image'}: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="*_0000.nii.gz image file")
    parser.add_argument("label", type=Path, help="*.nii.gz label file")
    parser.add_argument("--output-dir", type=Path, default=Path("augmented_previews"))
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of augmented patches to generate (default: 8)")
    parser.add_argument("--patch-size", type=int, nargs=3, default=[128, 384, 128])
    parser.add_argument("--num-classes", type=int, default=7)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref = sitk.ReadImage(str(args.image))
    spacing = ref.GetSpacing()

    # Use range normalization so values are [0,1] and look sensible in Slicer.
    # num_samples=1 so each transform call produces exactly one patch dict.
    transform = get_train_transforms(
        patch_size=tuple(args.patch_size),
        num_samples=1,
        num_classes=args.num_classes,
        normalization="range",
        clip_min=-1000.0,
        clip_max=500.0,
    )

    data = {"image": str(args.image), "label": str(args.label)}

    print(f"Source image : {args.image.name}")
    print(f"Source label : {args.label.name}")
    print(f"Patch size   : {args.patch_size}")
    print(f"Generating {args.n_samples} augmented patches → {args.output_dir.resolve()}\n")

    for i in range(args.n_samples):
        result = transform(data)
        # RandCropByLabelClassesd with num_samples=1 wraps output in a list
        if isinstance(result, list):
            result = result[0]

        img_np = _to_numpy(result["image"])
        lbl_np = _to_numpy(result["label"])

        print(f"[{i+1}/{args.n_samples}]")
        _save_nifti(img_np, spacing, args.output_dir / f"aug_{i:03d}_image.nii.gz", is_label=False)
        _save_nifti(lbl_np, spacing, args.output_dir / f"aug_{i:03d}_label.nii.gz", is_label=True)

    print(f"\nDone.")
    print("In 3D Slicer: File → Add Data")
    print("  aug_NNN_image.nii.gz  → load as  Volume")
    print("  aug_NNN_label.nii.gz  → load as  Segmentation")


if __name__ == "__main__":
    main()
