from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from monai.losses import DiceLoss

# Per-class CE weights (inverse-frequency based, capped at 10)
_COVERAGE_WEIGHTS: dict[int, float] = {
    1: 1.5,   # wood   (47.3%)
    2: 5.5,   # knot   (1.40%)
    3: 10.0,  # rot    (0.13%)
    4: 3.0,   # bark   (4.80%)
    5: 7.0,   # crack  (0.83%)
}

_CLASS_FREQS: list[float] = [
    0.454862,  # 0  background
    0.473440,  # 1  wood
    0.014032,  # 2  knot
    0.001256,  # 3  rot
    0.048018,  # 4  bark
    0.008318,  # 5  crack
    0.000074,  # 6  insect
]


def _build_weight_vector(
    num_classes: int,
    rare_label_idx: int,
    rare_class_weight: float,
    include_background: bool,
) -> torch.Tensor:
    """Return a normalized per-class weight tensor."""
    if include_background:
        weight_len = num_classes
    else:
        weight_len = max(1, num_classes - 1)

    weights = torch.ones(weight_len, dtype=torch.float32)

    for cls, w in _COVERAGE_WEIGHTS.items():
        if cls >= num_classes:
            continue
        idx = cls if include_background else cls - 1
        if cls == 0 and not include_background:
            continue
        if cls != rare_label_idx:
            weights[idx] = w

    if 0 <= rare_label_idx < num_classes:
        rare_idx = rare_label_idx if include_background else rare_label_idx - 1
        if 0 <= rare_idx < weight_len:
            weights[rare_idx] = rare_class_weight

    # Normalize so frequency-weighted mean = 1.0
    freq_values = _CLASS_FREQS[:num_classes] if include_background else _CLASS_FREQS[1:num_classes]
    if len(freq_values) == weight_len:
        freqs = torch.tensor(freq_values, dtype=torch.float32)
        fw_mean = float((weights * freqs).sum())
        if fw_mean > 0:
            weights = weights / fw_mean

    return weights


class _SkeletonRecallLoss(nn.Module):
    """1 - mean skeleton recall over foreground classes.

    Encourages the model to recover the topology of thin structures like cracks.
    Skeleton is computed non-differentiably; gradients flow only through probabilities.
    """

    def __init__(self, include_background: bool = False, epsilon: float = 1e-5):
        super().__init__()
        self.include_background = include_background
        self.epsilon = epsilon

    @torch.no_grad()
    def _compute_skeleton(self, onehot: torch.Tensor) -> torch.Tensor:
        from skimage.morphology import skeletonize

        B, C = onehot.shape[:2]
        arr = onehot.detach().cpu().numpy().astype(bool)
        out = np.zeros_like(arr, dtype=np.float32)
        for b in range(B):
            for c in range(C):
                if arr[b, c].any():
                    out[b, c] = skeletonize(arr[b, c]).astype(np.float32)
        return torch.from_numpy(out).to(onehot.device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  [B, C, D, H, W]
        # targets: [B, 1, D, H, W] integer class indices
        probs = torch.softmax(logits, dim=1)

        tgt = targets.long()
        if tgt.shape[1] == 1:
            tgt = tgt.squeeze(1)           # [B, D, H, W]

        onehot = torch.zeros_like(probs)
        onehot.scatter_(1, tgt.unsqueeze(1), 1.0)

        start = 0 if self.include_background else 1
        probs_fg  = probs[:, start:]
        onehot_fg = onehot[:, start:]

        skel = self._compute_skeleton(onehot_fg)  # [B, C_fg, ...]

        spatial = list(range(2, skel.ndim))
        numer = (probs_fg * skel).sum(dim=spatial)                 # [B, C_fg]
        denom = skel.sum(dim=spatial) + self.epsilon               # [B, C_fg]

        recall = numer / denom                                     # [B, C_fg]
        return 1.0 - recall.mean()


class CeDiceSkelLoss(nn.Module):
    """CE + Soft Dice + Skeleton Recall combined loss."""

    def __init__(
        self,
        num_classes: int = 7,
        rare_label_idx: int = 6,
        rare_class_weight: float = 10.0,
        lambda_ce: float = 0.5,
        lambda_dice: float = 1.0,
        lambda_skel: float = 1.0,
    ):
        super().__init__()
        self.lambda_ce   = lambda_ce
        self.lambda_dice = lambda_dice
        self.lambda_skel = lambda_skel

        ce_weights_fg = _build_weight_vector(
            num_classes, rare_label_idx, rare_class_weight, include_background=False
        )
        ce_weights = torch.cat([torch.ones(1), ce_weights_fg])
        self.register_buffer("ce_weights", ce_weights)

        self.ce = nn.CrossEntropyLoss(weight=self.ce_weights)

        self.dice = DiceLoss(
            to_onehot_y=True,
            softmax=True,
            include_background=False,
        )

        self.skel = _SkeletonRecallLoss(include_background=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tgt_ce = targets.squeeze(1).long()

        ce_loss   = self.ce(logits, tgt_ce)
        dice_loss = self.dice(logits, targets)
        skel_loss = self.skel(logits, targets)

        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss + self.lambda_skel * skel_loss


def _get_dice_focal_loss(
    num_classes: int,
    rare_label_idx: int,
    rare_class_weight: float,
    focal_gamma: float = 2.0,
):
    from monai.losses import DiceFocalLoss

    weights = _build_weight_vector(num_classes, rare_label_idx, rare_class_weight, include_background=False)

    return DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        gamma=focal_gamma,
        weight=weights,
        lambda_dice=0.5,
        lambda_focal=0.5,
    )


def get_loss(
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_weight: float = 10.0,
    loss_type: str = "combined",
) -> nn.Module:
    if loss_type == "combined":
        return CeDiceSkelLoss(
            num_classes=num_classes,
            rare_label_idx=rare_label_idx,
            rare_class_weight=rare_class_weight,
        )
    if loss_type == "dice_focal":
        return _get_dice_focal_loss(num_classes, rare_label_idx, rare_class_weight)
    raise ValueError(f"Unknown loss_type '{loss_type}'. Choose 'combined' or 'dice_focal'.")
