#!/usr/bin/env python3
"""Rule-based postprocessing for wood-defect segmentation predictions.

Label mapping (Dataset001_BPWoodDefects/dataset.json):
    0 = Background   1 = Healthy wood   2 = Knot   3 = Rot   4 = Bark
    5 = Crack        6 = Insect damage
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


LABEL_BACKGROUND = 0
LABEL_WOOD = 1
LABEL_KNOT = 2
LABEL_ROT = 3
LABEL_BARK = 4
LABEL_CRACK = 5
LABEL_INSECT = 6

DEFECT_LABELS = {LABEL_KNOT, LABEL_ROT, LABEL_BARK, LABEL_CRACK, LABEL_INSECT}


def _dilated(mask: np.ndarray, voxels: float) -> np.ndarray:
    """Return mask dilated by *voxels* (rounded to nearest int) using a 3D cross structure."""
    iterations = max(1, round(voxels))
    struct = ndimage.generate_binary_structure(mask.ndim, 1)
    return ndimage.binary_dilation(mask, structure=struct, iterations=iterations)


def apply_rot_to_crack_rule(
    prediction: np.ndarray,
    rot_crack_distance_vx: float = 2.0,
) -> np.ndarray:
    """Relabel rot components within rot_crack_distance_vx voxels of crack as crack."""
    rot_mask = prediction == LABEL_ROT
    crack_mask = prediction == LABEL_CRACK

    if not rot_mask.any() or not crack_mask.any():
        return prediction

    crack_neighborhood = _dilated(crack_mask, rot_crack_distance_vx)

    labeled_rot, n_components = ndimage.label(rot_mask)
    for comp_id in range(1, n_components + 1):
        component = labeled_rot == comp_id
        if (component & crack_neighborhood).any():
            prediction[component] = LABEL_CRACK

    return prediction


def apply_background_to_rot_rule(
    prediction: np.ndarray,
    bg_max_size_vx: int = 50_000,
) -> np.ndarray:
    """Relabel small background components adjacent to rot as rot.

    Size cap protects the large exterior background from being consumed.
    """
    bg_mask = prediction == LABEL_BACKGROUND
    rot_mask = prediction == LABEL_ROT

    if not bg_mask.any() or not rot_mask.any():
        return prediction

    rot_dilated = ndimage.binary_dilation(rot_mask)

    labeled_bg, n_components = ndimage.label(bg_mask)
    for comp_id in range(1, n_components + 1):
        component = labeled_bg == comp_id
        if int(component.sum()) > bg_max_size_vx:
            continue
        if (component & rot_dilated).any():
            prediction[component] = LABEL_ROT

    return prediction


def apply_crack_to_background_rule(
    prediction: np.ndarray,
    crack_bark_distance_vx: float = 10.0,
    crack_bark_fraction: float = 0.8,
) -> np.ndarray:
    """Relabel crack components where >= crack_bark_fraction of voxels are near bark as background."""
    crack_mask = prediction == LABEL_CRACK
    bark_mask = prediction == LABEL_BARK

    if not crack_mask.any() or not bark_mask.any():
        return prediction

    near_bark = _dilated(bark_mask, crack_bark_distance_vx)

    labeled_crack, n_components = ndimage.label(crack_mask)
    for comp_id in range(1, n_components + 1):
        component = labeled_crack == comp_id
        n_total = int(component.sum())
        n_near_bark = int((component & near_bark).sum())
        if n_near_bark / n_total >= crack_bark_fraction:
            prediction[component] = LABEL_BACKGROUND

    return prediction


def apply_enclosed_to_defect_rule(
    prediction: np.ndarray,
    max_hole_size_vx: int = 50_000,
) -> np.ndarray:
    """Fill HW/BG holes enclosed by a defect class in each axial slice."""
    is_hw_or_bg = (prediction == LABEL_BACKGROUND) | (prediction == LABEL_WOOD)

    for lbl in DEFECT_LABELS:
        defect_mask = prediction == lbl
        if not defect_mask.any():
            continue

        filled = defect_mask.copy()
        for z in range(defect_mask.shape[2]):
            filled[:, :, z] = ndimage.binary_fill_holes(defect_mask[:, :, z])

        holes = filled & ~defect_mask & is_hw_or_bg
        if not holes.any():
            continue

        labeled, n = ndimage.label(holes)
        for comp_id in range(1, n + 1):
            comp = labeled == comp_id
            if int(comp.sum()) <= max_hole_size_vx:
                prediction[comp] = lbl

    return prediction


def postprocess(
    prediction: np.ndarray,
    rot_crack_distance_vx: float = 2.0,
    crack_bark_distance_vx: float = 10.0,
    crack_bark_fraction: float = 0.8,
    bg_max_size_vx: int = 50_000,
    enclosed_max_hole_size_vx: int = 50_000,
) -> tuple[np.ndarray, dict[str, int]]:
    result = prediction.copy()

    apply_rot_to_crack_rule(result, rot_crack_distance_vx=rot_crack_distance_vx)
    apply_crack_to_background_rule(
        result,
        crack_bark_distance_vx=crack_bark_distance_vx,
        crack_bark_fraction=crack_bark_fraction,
    )
    apply_background_to_rot_rule(result, bg_max_size_vx=bg_max_size_vx)

    before_enclosed = result.copy()
    apply_enclosed_to_defect_rule(
        result,
        max_hole_size_vx=enclosed_max_hole_size_vx,
    )

    was_non_defect = (before_enclosed == LABEL_BACKGROUND) | (before_enclosed == LABEL_WOOD)
    is_now_defect  = ~((result == LABEL_BACKGROUND) | (result == LABEL_WOOD))

    stats = {
        "rot_to_crack":       int(((prediction == LABEL_ROT)        & (result == LABEL_CRACK)).sum()),
        "crack_to_bg":        int(((prediction == LABEL_CRACK)      & (result == LABEL_BACKGROUND)).sum()),
        "bg_to_rot":          int(((prediction == LABEL_BACKGROUND) & (result == LABEL_ROT)).sum()),
        "enclosed_to_defect": int((was_non_defect & is_now_defect).sum()),
    }
    return result, stats
