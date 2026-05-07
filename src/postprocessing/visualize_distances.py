#!/usr/bin/env python3
"""Pairwise class-distance matrix aggregated over all ground-truth labels.

For every (source, target) label pair, computes the mean and minimum 2D
per-slice Euclidean distance (voxels) from source voxels to the nearest
target voxel, aggregated across every slice of every NIfTI file in the
input directory (default: labelsTr).

Processing is done slice-by-slice so memory stays small regardless of
volume size.

Usage::

    # default: aggregate over all labelsTr files
    poetry run python -m src.postprocessing.visualize_distances

    # single file or custom directory
    poetry run python -m src.postprocessing.visualize_distances nnunet.nii.gz
    poetry run python -m src.postprocessing.visualize_distances path/to/labels/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

DEFAULT_LABELS_DIR = Path(
    "src/nn_UNet/nnunet_data/nnUNet_raw/Dataset001_BPWoodDefects/labelsTr"
)

CLASS_NAMES = ["Background", "Healthy Wood", "Knot", "Rot", "Bark", "Crack", "Insect"]
N = len(CLASS_NAMES)

CLASS_COLORS = ["#808080", "#DEB887", "#8B4513", "#228B22", "#D2691E", "#FF4500", "#FFD700"]


def _accumulate_volume(
    volume: np.ndarray,
    sum_dist: np.ndarray,
    counts: np.ndarray,
    min_dist: np.ndarray,
    max_dist: np.ndarray,
) -> None:
    """Add one volume's slice-by-slice distances into the running accumulators."""
    n_slices = volume.shape[2]
    for z in range(n_slices):
        sl = volume[:, :, z]
        for tgt in range(N):
            tgt_mask = sl == tgt
            if not tgt_mask.any():
                continue
            dist = distance_transform_edt(~tgt_mask)
            for src in range(N):
                if src == tgt:
                    continue
                src_mask = sl == src
                if not src_mask.any():
                    continue
                d = dist[src_mask]
                sum_dist[src, tgt] += d.sum()
                counts[src, tgt] += d.size
                if d.min() < min_dist[src, tgt]:
                    min_dist[src, tgt] = d.min()
                if d.max() > max_dist[src, tgt]:
                    max_dist[src, tgt] = d.max()


def compute_distance_matrix(
    files: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean_dist, min_dist, max_dist) matrices of shape (N, N).

    Entry [src, tgt] = mean / min / max 2D distance (voxels) from class *src*
    voxels to the nearest class *tgt* voxel, aggregated over all axial slices
    of all files where both classes co-occur.  Diagonal is 0.  NaN where the
    pair never co-occurs.
    """
    sum_dist = np.zeros((N, N), dtype=np.float64)
    counts = np.zeros((N, N), dtype=np.int64)
    min_dist = np.full((N, N), np.inf)
    max_dist = np.full((N, N), -np.inf)
    np.fill_diagonal(min_dist, 0.0)
    np.fill_diagonal(max_dist, 0.0)

    for path in tqdm(files, desc="Files", unit="file"):
        volume = np.asarray(nib.load(path).dataobj).astype(np.uint8)
        _accumulate_volume(volume, sum_dist, counts, min_dist, max_dist)

    with np.errstate(invalid="ignore"):
        mean_dist = np.where(counts > 0, sum_dist / counts, np.nan)

    np.fill_diagonal(mean_dist, 0.0)
    min_dist[np.isinf(min_dist)] = np.nan
    max_dist[np.isinf(max_dist)] = np.nan

    return mean_dist, min_dist, max_dist


def _make_annot(data: np.ndarray) -> np.ndarray:
    out = np.empty_like(data, dtype=object)
    for i in range(N):
        for j in range(N):
            if i == j:
                out[i, j] = "-"
            elif np.isnan(data[i, j]):
                out[i, j] = ""
            else:
                v = data[i, j]
                out[i, j] = f"{v:.1f}" if v < 100 else f"{v:.0f}"
    return out


def _sym_mean(mean_dist: np.ndarray) -> np.ndarray:
    """Symmetric mean: (d[A->B] + d[B->A]) / 2.  If one direction is NaN use the other."""
    a, b = mean_dist.copy(), mean_dist.T.copy()
    return np.where(np.isnan(a), b, np.where(np.isnan(b), a, (a + b) / 2))


def _draw_heatmap(ax, data: np.ndarray, title: str) -> None:
    GREY = "#cccccc"
    no_data = np.isnan(data) | np.eye(N, dtype=bool)
    data_plot = np.where(no_data, 0.0, data)

    sns.heatmap(
        data_plot,
        ax=ax,
        mask=no_data,
        annot=_make_annot(data),
        fmt="",
        cmap="YlOrRd",
        vmin=0,
        linewidths=0.6,
        linecolor="white",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "distance (voxels)", "shrink": 0.8},
        annot_kws={"size": 9, "weight": "bold"},
    )

    for i in range(N):
        for j in range(N):
            if no_data[i, j]:
                ax.add_patch(mpatches.Rectangle((j, i), 1, 1, fill=True, facecolor=GREY, lw=0))

    for i in range(N):
        ax.text(i + 0.5, i + 0.5, "-", ha="center", va="center", fontsize=9, color="#555555", fontweight="bold")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Class B", fontsize=11)
    ax.set_ylabel("Class A", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)


def plot(mean_dist: np.ndarray, out_path: Path) -> None:
    sym = _sym_mean(mean_dist)

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    fig.suptitle("Pairwise class distance matrix", fontsize=16, fontweight="bold", y=1.01)

    _draw_heatmap(axes[0], mean_dist, "Asymmetric mean (vx)\n[row = source,  col = target]")
    axes[0].set_xlabel("Target class", fontsize=11)
    axes[0].set_ylabel("Source class", fontsize=11)

    _draw_heatmap(axes[1], sym, "Symmetric mean (vx)\n(d[A→B] + d[B→A]) / 2")
    axes[1].set_xlabel("Class B", fontsize=11)
    axes[1].set_ylabel("Class A", fontsize=11)

    legend_patches = [mpatches.Patch(facecolor="#cccccc", label="no co-occurrence in any slice")]
    fig.legend(handles=legend_patches, loc="lower center", ncol=1, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def _fmt_matrix(m: np.ndarray) -> str:
    header = f"{'':>12}" + "".join(f"{n:>12}" for n in CLASS_NAMES)
    rows = [header]
    for i, row_name in enumerate(CLASS_NAMES):
        cells = [f"{'—':>12}" if np.isnan(m[i, j]) else f"{m[i, j]:>12.1f}" for j in range(N)]
        rows.append(f"{row_name:>12}" + "".join(cells))
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Pairwise class-distance heatmap aggregated over ground-truth label files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_LABELS_DIR,
        help="NIfTI file (.nii.gz) or directory of NIfTI files. "
             f"Defaults to {DEFAULT_LABELS_DIR}",
    )
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path (default: <input_dir>/distances.png or <stem>_distances.png).")
    args = p.parse_args(argv)

    inp: Path = args.input

    if inp.is_dir():
        files = sorted(inp.glob("*.nii.gz"))
        if not files:
            p.error(f"No *.nii.gz files found in {inp}")
        out = args.out or inp / "distances.png"
        cache = inp / "distances_mean.npy"
        print(f"Aggregating over {len(files)} files in {inp}")
    else:
        files = [inp]
        stem = inp.name[:-len(".nii.gz")] if inp.name.endswith(".nii.gz") else inp.stem
        out = args.out or inp.parent / f"{stem}_distances.png"
        cache = inp.parent / f"{stem}_distances_mean.npy"
        print(f"Processing {inp}")

    if cache.exists() and not files[0].exists():
        print(f"NIfTI not found — loading cached distances from {cache}")
        mean_dist = np.load(cache)
    else:
        for f in files:
            print(f"  {f.name}")
        mean_dist, min_dist, max_dist = compute_distance_matrix(files)
        np.save(cache, mean_dist)
        print(f"Cached distances to {cache}")

    print("\nMean distance matrix (vx) — asymmetric, average over co-occurring slices:")
    print(_fmt_matrix(mean_dist))
    print("\nSymmetric mean (vx) — (d[A->B] + d[B->A]) / 2:")
    print(_fmt_matrix(_sym_mean(mean_dist)))

    plot(mean_dist, out)


if __name__ == "__main__":
    main()
