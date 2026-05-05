#!/usr/bin/env python3
"""Compute and print row-normalised confusion matrices for all three models.

Run on the cluster after postprocessing:

    python -m src.postprocessing.confusion_matrix

Paths match the prediction files recorded in summary.json / *_pp.json.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib

GT_PATH   = "/home/morgaoli/dub3_gt/dub_3_first500.nii.gz"
MODELS = {
    "nnU-Net":    "/home/morgaoli/dub3_out_pp/dub_3_first500.nii.gz",
    "MedNeXt":    "/home/morgaoli/eval_tmp/mednext_pp_r/dub_3_first500.nii.gz",
    "SwinUNETR":  "/home/morgaoli/eval_tmp/swinunetr_pp_r/dub_3_first500.nii.gz",
}
CLASS_NAMES = ["BG", "HW", "Knot", "Rot", "Bark", "Crack", "Insect"]
N = len(CLASS_NAMES)


def load(path: str) -> np.ndarray:
    return np.asarray(nib.load(path).dataobj, dtype=np.int32).ravel()


def confusion(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    mat = np.zeros((N, N), dtype=np.int64)
    for t in range(N):
        mask = gt == t
        for p in range(N):
            mat[t, p] = int(np.sum(pred[mask] == p))
    return mat


def to_latex(mat: np.ndarray, model: str) -> str:
    row_sums = mat.sum(axis=1, keepdims=True).clip(min=1)
    pct = (mat / row_sums * 100).round(1)

    col_header = " & ".join(f"\\textbf{{{c}}}" for c in CLASS_NAMES)
    lines = [
        f"% Confusion matrix — {model}",
        "\\begin{table}[H]",
        "    \\centering",
        f"    \\caption{{Row-normalised confusion matrix for {model} on the held-out test log \\textit{{dub3}}. "
        "Values are percentages of each true class. The diagonal (bold) shows correctly classified voxels.}}",
        f"    \\label{{tab:cm_{model.lower().replace('-','').replace(' ','')}}}",
        "    \\begin{adjustbox}{max width=\\textwidth}",
        "    \\begin{tabular}{l" + "r" * N + "}",
        "        \\toprule",
        f"        & {col_header} \\\\",
        "        \\midrule",
    ]
    for i, row_name in enumerate(CLASS_NAMES):
        cells = []
        for j in range(N):
            val = f"{pct[i, j]:.1f}"
            cells.append(f"\\textbf{{{val}}}" if i == j else val)
        lines.append(f"        {row_name} & {' & '.join(cells)} \\\\")
    lines += [
        "        \\bottomrule",
        "    \\end{tabular}",
        "    \\end{adjustbox}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def main() -> None:
    gt = load(GT_PATH)
    for model, path in MODELS.items():
        pred = load(path)
        mat  = confusion(gt, pred)
        print(to_latex(mat, model))
        print()


if __name__ == "__main__":
    main()
