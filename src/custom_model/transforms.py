from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandAffined,
    RandAdjustContrastd,
    RandCropByLabelClassesd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    Rand3DElasticd,
    RandCoarseDropoutd,
    RandHistogramShiftd,
    RandZoomd,
    ScaleIntensityRanged,
    SpatialPadd,
)

# Patch sampling bias per class — higher = more crops centred on that class
_MINORITY_CROP_RATIOS: dict[int, float] = {
    1: 2.0,   # wood
    2: 4.0,   # knot   (1.40%)
    3: 6.0,   # rot    (0.13%, hardest to hit randomly)
    4: 2.0,   # bark   (4.80%)
    5: 5.0,   # crack  (0.83%, elongated)
}


def _build_intensity_normalization_transforms(
    normalization: str,
    clip_min: float,
    clip_max: float,
    zscore_mean: float | None,
    zscore_std: float | None,
) -> list:
    if normalization == "range":
        return [
            ScaleIntensityRanged(
                keys="image",
                a_min=clip_min,
                a_max=clip_max,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            )
        ]

    if normalization == "zscore":
        if zscore_mean is None or zscore_std is None:
            raise ValueError("zscore normalization requires zscore_mean and zscore_std")
        if zscore_std <= 0:
            raise ValueError("zscore_std must be > 0")
        return [
            ScaleIntensityRanged(
                keys="image",
                a_min=clip_min,
                a_max=clip_max,
                b_min=clip_min,
                b_max=clip_max,
                clip=True,
            ),
            NormalizeIntensityd(
                keys="image",
                subtrahend=float(zscore_mean),
                divisor=float(zscore_std),
                nonzero=False,
                channel_wise=False,
            ),
        ]

    raise ValueError(f"Unsupported normalization mode: {normalization}")


def get_train_transforms(
    patch_size,
    num_samples,
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_oversample: int = 8,
    normalization: str = "range",
    clip_min: float = -1000.0,
    clip_max: float = 500.0,
    zscore_mean: float | None = None,
    zscore_std: float | None = None,
):
    # Coarse dropout needs to fill with the normalized air value, not 0.
    # After z-score: air = (clip_min - mean) / std.  After range: air = 0.0.
    if normalization == "zscore" and zscore_mean is not None and zscore_std is not None:
        _coarse_fill = float((clip_min - zscore_mean) / zscore_std)
    else:
        _coarse_fill = 0.0

    ratios = [1.0] * num_classes
    for cls, r in _MINORITY_CROP_RATIOS.items():
        if cls < num_classes and cls != rare_label_idx:
            ratios[cls] = r
    if 0 <= rare_label_idx < num_classes:
        ratios[rare_label_idx] = float(rare_class_oversample)

    intensity_transforms = _build_intensity_normalization_transforms(
        normalization=normalization,
        clip_min=clip_min,
        clip_max=clip_max,
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
    )

    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS", labels=None),
            *intensity_transforms,
            SpatialPadd(keys=("image", "label"), spatial_size=patch_size),
            RandCropByLabelClassesd(
                keys=("image", "label"),
                label_key="label",
                spatial_size=patch_size,
                ratios=ratios,
                num_classes=num_classes,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
                warn=False,
            ),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
            # axis=1 is trunk length — flipping inverts growth direction
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=2),
            Rand3DElasticd(
                keys=("image", "label"),
                mode=("bilinear", "nearest"),
                prob=0.2,
                sigma_range=(5, 6),
                magnitude_range=(10, 60),
                padding_mode="reflection",
            ),
            RandAffined(
                keys=("image", "label"),
                mode=("bilinear", "nearest"),
                prob=0.25,
                rotate_range=(0.52, 0.52, 0.52),
                shear_range=(0.05, 0.05, 0.05),
                translate_range=(10, 10, 10),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="reflection",
            ),
            RandZoomd(
                keys=("image", "label"),
                min_zoom=0.5,
                max_zoom=1.0,
                mode=("trilinear", "nearest"),
                prob=0.25,
                keep_size=True,
            ),
            RandCoarseDropoutd(
                keys="image",
                holes=5,
                spatial_size=(16, 16, 16),
                prob=0.15,
                fill_value=_coarse_fill,
            ),
            RandHistogramShiftd(keys="image", num_control_points=4, prob=0.05),
            RandGaussianNoised(keys="image", prob=0.1, std=0.1),
            RandGaussianSharpend(
                keys="image",
                sigma1_x=(0.5, 1.0),
                sigma1_y=(0.5, 1.0),
                sigma1_z=(0.5, 1.0),
                prob=0.08,
            ),
            RandGaussianSmoothd(
                keys="image",
                sigma_x=(0.3, 0.8),
                sigma_y=(0.3, 0.8),
                sigma_z=(0.3, 0.8),
                prob=0.08,
            ),
            RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),
            RandAdjustContrastd(keys="image", prob=0.1, gamma=(0.7, 1.5), invert_image=True),
            RandScaleIntensityd(keys="image", factors=0.08, prob=0.12),
            EnsureTyped(keys=("image", "label")),
        ]
    )


def get_val_transforms(
    normalization: str = "range",
    clip_min: float = -1000.0,
    clip_max: float = 500.0,
    zscore_mean: float | None = None,
    zscore_std: float | None = None,
):
    intensity_transforms = _build_intensity_normalization_transforms(
        normalization=normalization,
        clip_min=clip_min,
        clip_max=clip_max,
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
    )

    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS", labels=None),
            *intensity_transforms,
            EnsureTyped(keys=("image", "label")),
        ]
    )


def get_inference_transforms(
    normalization: str = "range",
    clip_min: float = -1000.0,
    clip_max: float = 500.0,
    zscore_mean: float | None = None,
    zscore_std: float | None = None,
):
    """Image-only transforms for inference (no label key required)."""
    intensity_transforms = _build_intensity_normalization_transforms(
        normalization=normalization,
        clip_min=clip_min,
        clip_max=clip_max,
        zscore_mean=zscore_mean,
        zscore_std=zscore_std,
    )

    return Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            *intensity_transforms,
            EnsureTyped(keys=["image"]),
        ]
    )
