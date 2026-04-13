#!/usr/bin/env python3
"""
Label statistics for the nnU-Net raw dataset.

For every segmentation class reports:
  - Presence rate  : % of volumes that contain the class
  - Volume fraction: % of total voxels belonging to the class
  - Instance count : 3-D connected-component count (avg / volume)
  - Instance volume: median / min / max component size (voxels)
  - 2-D shape (sampled from cross-sectional slices that contain the class):
      aspect ratio  -- minor/major axis of fitted ellipse (1=square, 0=very elongated)
      roundness     -- circularity = 4π·area/perimeter²  (1=circle, 0=irregular/spiky)

Usage
-----
  # default: labelsTr inside the standard nnUNet_raw folder
  python src/nn_UNet/label_stats.py

  # custom directory (e.g. on cluster)
  python src/nn_UNet/label_stats.py /path/to/labelsTr

  # limit number of files processed (quick sanity check)
  python src/nn_UNet/label_stats.py --max-files 4

  # save results to CSV
  python src/nn_UNet/label_stats.py --csv stats.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

# ── project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── label map — index → name, must match the voxel values in your .nii.gz files ──
# Raw dataset.json has {"name": index}, so this is the inverted version.
LABEL_NAMES: Dict[int, str] = {
    0: "pozadi",
    1: "zdrave_drevo",
    2: "suk",
    3: "hniloba",
    4: "kura",
    5: "trhlina",
    6: "poskozeni_hmyzem",
}

DEFAULT_LABELS_DIR = (
    PROJECT_ROOT
    / "src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/labelsTr"
)

# ── helpers ───────────────────────────────────────────────────────────────────

def load_volume(path: Path) -> np.ndarray:
    """Return label volume as (Z, H, W) uint8 numpy array."""
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img).astype(np.uint8)


def _shape_stats_from_slice(binary_slice: np.ndarray) -> List[dict]:
    """
    Return per-component 2-D shape stats from a single binary (H×W) slice.

    Uses OpenCV for speed; each component yields:
      area, aspect_ratio (minor/major, 0-1), roundness (0-1).
    """
    uint8 = binary_slice.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uint8, connectivity=8)
    results = []
    for comp_id in range(1, n_labels):          # skip background (id=0)
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        if area < 4:                              # ignore sub-pixel noise
            continue

        comp_mask = (labels == comp_id).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        perimeter = cv2.arcLength(contours[0], closed=True)
        roundness = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 1.0

        # Fitted ellipse gives major / minor axes (needs ≥ 5 points)
        if len(contours[0]) >= 5:
            (_, _), (minor, major), _ = cv2.fitEllipse(contours[0])
            aspect_ratio = (minor / major) if major > 0 else 1.0
        else:
            # Fall back to bounding-box ratio
            w = stats[comp_id, cv2.CC_STAT_WIDTH]
            h = stats[comp_id, cv2.CC_STAT_HEIGHT]
            aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 1.0

        results.append({
            "area":         area,
            "aspect_ratio": float(np.clip(aspect_ratio, 0.0, 1.0)),
            "roundness":    float(np.clip(roundness, 0.0, 1.0)),
        })
    return results


# How many 2-D slices to sample per volume per class (keeps runtime reasonable)
_MAX_SLICES_PER_VOLUME = 30


def analyse_volume(
    vol: np.ndarray,
) -> Dict[int, dict]:
    """
    Compute per-class statistics for a single (Z, H, W) label volume.

    Returns a dict keyed by class index with fields:
      voxel_count, n_components_3d, shape_samples (list of dicts)
    """
    results: Dict[int, dict] = {}

    for cls in LABEL_NAMES:
        binary = (vol == cls)
        voxel_count = int(binary.sum())
        if voxel_count == 0:
            results[cls] = {"voxel_count": 0, "n_components_3d": 0, "shape_samples": []}
            continue

        # 3-D connected components (background class skipped for speed)
        if cls == 0:
            n_comp = 0
        else:
            _, n_comp = ndi.label(binary)

        # 2-D shape sampling: only slices that contain the class
        shape_samples: List[dict] = []
        if cls != 0:
            z_indices = np.where(binary.any(axis=(1, 2)))[0]
            if len(z_indices) > _MAX_SLICES_PER_VOLUME:
                rng = np.random.default_rng(seed=42)
                z_indices = rng.choice(z_indices, _MAX_SLICES_PER_VOLUME, replace=False)
            for z in z_indices:
                shape_samples.extend(_shape_stats_from_slice(binary[z]))

        results[cls] = {
            "voxel_count":      voxel_count,
            "n_components_3d":  n_comp,
            "shape_samples":    shape_samples,
        }

    return results


# ── main analysis ─────────────────────────────────────────────────────────────

def analyse_dataset(labels_dir: Path, max_files: int | None = None) -> List[dict]:
    """
    Analyse all *.nii.gz files in labels_dir.

    Returns a list of per-class summary dicts for optional CSV export.
    """
    files = sorted(labels_dir.glob("*.nii.gz"))
    if not files:
        print(f"No .nii.gz files found in {labels_dir}")
        return []
    if max_files:
        files = files[:max_files]

    n_files = len(files)
    print(f"Analysing {n_files} label file(s) in:\n  {labels_dir}\n")

    # Accumulators
    presence:       Dict[int, int]         = defaultdict(int)
    total_voxels:   int                    = 0
    class_voxels:   Dict[int, int]         = defaultdict(int)
    n_comp_per_vol: Dict[int, List[int]]   = defaultdict(list)
    all_shapes:     Dict[int, List[dict]]  = defaultdict(list)

    for idx, path in enumerate(files, 1):
        print(f"  [{idx:3d}/{n_files}] {path.name}", end="\r", flush=True)
        vol = load_volume(path)
        total_voxels += vol.size

        per_class = analyse_volume(vol)
        for cls, data in per_class.items():
            class_voxels[cls] += data["voxel_count"]
            if data["voxel_count"] > 0:
                presence[cls] += 1
            if cls != 0:
                n_comp_per_vol[cls].append(data["n_components_3d"])
                all_shapes[cls].extend(data["shape_samples"])

    print()  # clear \r line

    # -- print summary table --
    hdr = (
        f"{'Class':<25} {'Present':>9} {'Coverage':>10} "
        f"{'AvgComps':>10} {'AvgVol':>9} {'AvgAspect':>10} {'AvgRound':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows: List[dict] = []
    for cls, name in LABEL_NAMES.items():
        pct_pres = 100 * presence[cls] / n_files
        pct_cov  = 100 * class_voxels[cls] / total_voxels if total_voxels else 0.0

        comps = n_comp_per_vol[cls]
        shapes = all_shapes[cls]

        avg_comps  = np.mean(comps)    if comps  else 0.0
        avg_area   = np.mean([s["area"]         for s in shapes]) if shapes else 0.0
        avg_aspect = np.mean([s["aspect_ratio"] for s in shapes]) if shapes else 0.0
        avg_round  = np.mean([s["roundness"]    for s in shapes]) if shapes else 0.0

        print(
            f"{name:<25} {pct_pres:>8.1f}%  {pct_cov:>9.3f}%  "
            f"{avg_comps:>10.1f}  {avg_area:>9.1f}  {avg_aspect:>10.4f}  {avg_round:>10.4f}"
        )
        rows.append({
            "class_idx":      cls,
            "class_name":     name,
            "presence_pct":   round(pct_pres, 2),
            "coverage_pct":   round(pct_cov, 4),
            "avg_comps_3d":   round(avg_comps, 2),
            "avg_2d_area_px": round(avg_area, 1),
            "avg_aspect":     round(avg_aspect, 4),
            "avg_roundness":  round(avg_round, 4),
        })

    # -- detailed per-class breakdown --
    for cls, name in LABEL_NAMES.items():
        if cls == 0:
            continue
        shapes = all_shapes[cls]
        comps  = n_comp_per_vol[cls]
        print(f"\n  -- {name} --")
        print(f"     Present in {presence[cls]}/{n_files} volumes "
              f"({100*presence[cls]/n_files:.1f} %)")
        if comps:
            print(f"     3-D instances/volume : "
                  f"mean={np.mean(comps):.1f}  median={np.median(comps):.0f}  "
                  f"max={max(comps)}  total={sum(comps)}")
        if shapes:
            areas   = [s["area"]         for s in shapes]
            aspects = [s["aspect_ratio"] for s in shapes]
            rounds  = [s["roundness"]    for s in shapes]
            print(f"     2-D area (px)  : "
                  f"min={min(areas)}  median={int(np.median(areas))}  "
                  f"max={max(areas)}  mean={np.mean(areas):.1f}")
            print(f"     Aspect ratio   : "
                  f"min={min(aspects):.3f}  median={np.median(aspects):.3f}  "
                  f"max={max(aspects):.3f}  "
                  f"(1=square / circular, 0=very elongated)")
            print(f"     Roundness      : "
                  f"min={min(rounds):.3f}  median={np.median(rounds):.3f}  "
                  f"max={max(rounds):.3f}  "
                  f"(1=circle, 0=spiky/irregular)")
        else:
            print("     No 2-D shape samples collected.")

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="nnU-Net label statistics: coverage, roundness, aspect ratio"
    )
    parser.add_argument(
        "labels_dir",
        nargs="?",
        default=str(DEFAULT_LABELS_DIR),
        help="Path to the labelsTr directory (default: %(default)s)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help="Only process first N files (useful for a quick check)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        metavar="FILE",
        help="Save summary table to a CSV file",
    )
    args = parser.parse_args()

    rows = analyse_dataset(Path(args.labels_dir), max_files=args.max_files)

    if args.csv and rows:
        out = Path(args.csv)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSummary saved to {out}")


if __name__ == "__main__":
    main()
