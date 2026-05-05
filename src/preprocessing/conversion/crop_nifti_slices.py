#!/usr/bin/env python3
"""Crop a NIfTI volume to the first N slices along the Z axis."""

import argparse
import SimpleITK as sitk


def main():
    parser = argparse.ArgumentParser(description="Crop NIfTI to first N slices.")
    parser.add_argument("input", help="Input .nii / .nii.gz path")
    parser.add_argument("output", help="Output .nii / .nii.gz path")
    parser.add_argument("--slices", type=int, default=500, help="Number of slices to keep (default: 500)")
    args = parser.parse_args()

    img = sitk.ReadImage(args.input)
    print(f"Input size:  {img.GetSize()}")

    cropped = img[:, :, : args.slices]
    sitk.WriteImage(cropped, args.output)
    print(f"Output size: {cropped.GetSize()}  →  {args.output}")


if __name__ == "__main__":
    main()
