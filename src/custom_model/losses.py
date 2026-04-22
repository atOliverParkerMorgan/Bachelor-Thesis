import torch
from monai.losses import DiceFocalLoss

# Focal weights derived from stats.csv voxel coverage using sqrt(ref/coverage),
# capped at 10 to prevent gradient instability.
# Formula: min(10, sqrt(47.344 / class_coverage_pct))
#
#   class 2  suk/knot       1.4032% → sqrt(33.74) = 5.8  → 5.5
#   class 3  Hniloba/rot    0.1256% → sqrt(376.9) = 19.4 → 10.0 (capped)
#   class 4  Kůra/bark      4.8018% → sqrt(9.86)  = 3.1  → 3.0
#   class 5  Trhlina/crack  0.8318% → sqrt(56.9)  = 7.5  → 7.0
#   class 6  Poškození hm.  0.0074% → handled by rare_class_weight (30)
_COVERAGE_WEIGHTS: dict[int, float] = {
    2: 5.5,   # suk / knot       (1.40%)
    3: 10.0,  # Hniloba / rot    (0.13%) — tiny blobs, 108 px avg area
    4: 3.0,   # Kůra / bark      (4.80%)
    5: 7.0,   # Trhlina / crack  (0.83%) — elongated, aspect 0.18
}

# Voxel-frequency fractions from stats.csv (classes 0–6).
# Used to normalize focal weights so the frequency-weighted mean stays at 1.0,
# preventing loss inflation when minority-oversampled patches dominate batches.
_CLASS_FREQS: list[float] = [
    0.454862,  # class 0  background
    0.473440,  # class 1  wood (majority)
    0.014032,  # class 2  suk/knot
    0.001256,  # class 3  Hniloba/rot
    0.048018,  # class 4  Kůra/bark
    0.008318,  # class 5  Trhlina/crack
    0.000074,  # class 6  Poškození hmyzem
]


def get_loss(
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_weight: float = 15.0,
    focal_gamma: float = 2.0,
) -> DiceFocalLoss:
    """Return a DiceFocal loss with coverage-based per-class weights.

    Focal modulation (1-p)^gamma pushes gradient signal toward hard/uncertain
    voxels — beneficial for tiny structures (rot 108px, insect 68px) where the
    model is most uncertain early in training.  Per-class weights still handle
    the frequency imbalance; focal gamma handles the confidence imbalance.
    """
    include_background = False

    # MONAI expects weight length to match the effective class count used by
    # Dice/Focal. With include_background=False this is num_classes - 1.
    if include_background:
        weight_len = num_classes
    else:
        weight_len = max(1, num_classes - 1)

    focal_weights = torch.ones(weight_len, dtype=torch.float32)

    for cls, w in _COVERAGE_WEIGHTS.items():
        if cls >= num_classes:
            continue
        if include_background:
            idx = cls
        else:
            if cls == 0:
                continue
            idx = cls - 1
        if cls != rare_label_idx:
            focal_weights[idx] = w

    if 0 <= rare_label_idx < num_classes:
        if include_background:
            rare_idx = rare_label_idx
        else:
            rare_idx = rare_label_idx - 1
        if 0 <= rare_idx < weight_len:
            focal_weights[rare_idx] = rare_class_weight

    # Normalize so the frequency-weighted mean equals 1.0.
    if include_background:
        freq_values = _CLASS_FREQS[:num_classes]
    else:
        freq_values = _CLASS_FREQS[1:num_classes]

    if len(freq_values) == weight_len:
        freqs = torch.tensor(freq_values, dtype=torch.float32)
        fw_mean = float((focal_weights * freqs).sum())
        if fw_mean > 0:
            focal_weights = focal_weights / fw_mean

    return DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=include_background,
        gamma=focal_gamma,
        weight=focal_weights,
        lambda_dice=0.5,
        lambda_focal=0.5,
    )
