import inspect

import torch
from monai.losses import DiceCELoss

# CE weights derived from stats.csv voxel coverage using sqrt(ref/coverage),
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


def get_loss(
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_weight: float = 15.0,
) -> DiceCELoss:
    """Return a DiceCE loss with coverage-based CE weights for all minority classes.

    All foreground classes receive inverse-log-frequency weights so
    rare structures (rot, crack, …) contribute proportionally more to the
    CE component.  The rarest class (``rare_label_idx``) keeps its own
    dedicated ``rare_class_weight`` multiplier on top.
    """
    ce_weights = torch.ones(num_classes)
    for cls, w in _COVERAGE_WEIGHTS.items():
        if cls < num_classes and cls != rare_label_idx:
            ce_weights[cls] = w
    ce_weights[rare_label_idx] = rare_class_weight

    kwargs = dict(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )

    # MONAI changed CE-weight keyword names across versions. Pick the one
    # supported by the installed DiceCELoss to avoid runtime crashes.
    param_names = inspect.signature(DiceCELoss.__init__).parameters
    if "ce_weight" in param_names:
        kwargs["ce_weight"] = ce_weights
    elif "ce_weights" in param_names:
        kwargs["ce_weights"] = ce_weights
    elif "weight" in param_names:
        kwargs["weight"] = ce_weights

    return DiceCELoss(**kwargs)
