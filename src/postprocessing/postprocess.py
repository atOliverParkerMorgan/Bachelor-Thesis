#!/usr/bin/env python3
"""CLI for rule-based postprocessing of wood-defect segmentation predictions.

Applies four spatial rules to a nnU-Net / MedNext / SwinUNETR NIfTI prediction:

  Rule 1 – Rot → Crack
      Any rot connected component whose closest voxel is within
      --rot-crack-dist voxels of any crack voxel is relabelled as crack.

  Rule 2 – Crack → Background
      Any crack voxel within --crack-bark-dist voxels of bark is relabelled as
      background.

  Rule 3 – Background → Rot
      Any background connected component smaller than --bg-max-size voxels that
      is directly adjacent to rot is relabelled as rot.

  Rule 4 – Enclosed HW/BG → Defect
      Healthy-wood or background holes enclosed by a single defect class
      (per axial slice) and smaller than --enclosed-max-hole-size voxels are
      filled with that defect label.

Usage – single file::

    poetry run python -m src.postprocessing.postprocess \\
        path/to/pred.nii.gz path/to/pred_pp.nii.gz

Usage – whole directory (processes every *.nii.gz)::

    poetry run python -m src.postprocessing.postprocess \\
        src/nn_UNet/predictions/ src/nn_UNet/predictions_postprocessed/

All distance parameters are in voxels (isotropic assumption).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from src.postprocessing.rules import postprocess


def _process_file(
    src: Path,
    dst: Path,
    rot_crack_distance_vx: float,
    crack_bark_distance_vx: float,
    crack_bark_fraction: float,
    bg_max_size_vx: int,
    enclosed_max_hole_size_vx: int = 50_000,
) -> dict[str, int]:
    img = nib.load(src)
    data = np.asarray(img.dataobj).astype(np.int16)

    result, stats = postprocess(
        data,
        rot_crack_distance_vx=rot_crack_distance_vx,
        crack_bark_distance_vx=crack_bark_distance_vx,
        crack_bark_fraction=crack_bark_fraction,
        bg_max_size_vx=bg_max_size_vx,
        enclosed_max_hole_size_vx=enclosed_max_hole_size_vx,
    )

    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, img.affine, img.header), dst)
    return stats


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rule-based postprocessing for wood-defect segmentation predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", type=Path, help="Input NIfTI file or directory of *.nii.gz files.")
    p.add_argument("output", type=Path, help="Output NIfTI file or output directory.")
    p.add_argument(
        "--rot-crack-dist",
        type=float,
        default=2.0,
        metavar="VX",
        help="Max distance (voxels) from crack for a rot component to be relabelled as crack. "
             "1–2 means touching/directly adjacent; increase slightly for 'very close'.",
    )
    p.add_argument(
        "--bg-max-size",
        type=int,
        default=50_000,
        metavar="VX",
        help="Background connected components larger than this (voxels) are never relabelled "
             "as rot, protecting the main exterior background.",
    )
    p.add_argument(
        "--crack-bark-dist",
        type=float,
        default=10.0,
        metavar="VX",
        help="Distance threshold (voxels) used to decide whether a crack voxel is 'near bark'.",
    )
    p.add_argument(
        "--crack-bark-frac",
        type=float,
        default=0.8,
        metavar="FRAC",
        help="Minimum fraction [0–1] of a crack component's voxels that must be near bark "
             "for the whole component to be relabelled as background.",
    )
    p.add_argument(
        "--enclosed-max-hole-size",
        type=int,
        default=50_000,
        metavar="VX",
        help="Max size (voxels) of a HW or BG hole that can be filled by the surrounding "
             "defect. Larger components are left untouched.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    inp: Path = args.input
    out: Path = args.output

    if inp.is_dir():
        nii_files = sorted(inp.glob("*.nii.gz"))
        if not nii_files:
            print(f"No *.nii.gz files found in {inp}", file=sys.stderr)
            sys.exit(1)

        total_rot_to_crack = 0
        total_crack_to_bg = 0
        for src in nii_files:
            dst = out / src.name
            stats = _process_file(
                src, dst,
                rot_crack_distance_vx=args.rot_crack_dist,
                crack_bark_distance_vx=args.crack_bark_dist,
                crack_bark_fraction=args.crack_bark_frac,
                bg_max_size_vx=args.bg_max_size,
                enclosed_max_hole_size_vx=args.enclosed_max_hole_size,
            )
            total_rot_to_crack += stats["rot_to_crack"]
            total_crack_to_bg += stats["crack_to_bg"]
            print(
                f"  {src.name}: rot->crack {stats['rot_to_crack']:,} vx, "
                f"bg->rot {stats['bg_to_rot']:,} vx, "
                f"crack->bg {stats['crack_to_bg']:,} vx, "
                f"enclosed->defect {stats['enclosed_to_defect']:,} vx"
            )

        print(f"\nDone ({len(nii_files)} files).")

    elif inp.is_file():
        if out.is_dir():
            out = out / inp.name
        stats = _process_file(
            inp, out,
            rot_crack_distance_vx=args.rot_crack_dist,
            crack_bark_distance_vx=args.crack_bark_dist,
            crack_bark_fraction=args.crack_bark_frac,
            bg_max_size_vx=args.bg_max_size,
            enclosed_max_hole_size_vx=args.enclosed_max_hole_size,
        )
        print(
            f"Saved {out}\n"
            f"  rot->crack:        {stats['rot_to_crack']:,} vx\n"
            f"  bg->rot:           {stats['bg_to_rot']:,} vx\n"
            f"  crack->bg:         {stats['crack_to_bg']:,} vx\n"
            f"  enclosed->defect:  {stats['enclosed_to_defect']:,} vx"
        )

    else:
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
