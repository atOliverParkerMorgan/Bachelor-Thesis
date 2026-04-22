from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
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
    ScaleIntensityRanged,
    SpatialPadd,
)

# Crop ratios: min(8, sqrt(47.344 / class_coverage_pct)), same formula as
# loss weights so sampling focus matches gradient signal.
#   class 2  knot  1.40% → 5.8 → 4.0 (capped lower — knots are large enough)
#   class 3  rot   0.13% → 19  → 6.0 (capped — tiny 108 px blobs)
#   class 4  bark  4.80% → 3.1 → 2.0 (large area, easier to hit randomly)
#   class 5  crack 0.83% → 7.5 → 5.0 (elongated — needs explicit centering)
_MINORITY_CROP_RATIOS: dict[int, float] = {
    2: 4.0,   # suk / knot       (1.40%, 517 px avg area)
    3: 6.0,   # Hniloba / rot    (0.13%, 108 px avg area — hardest to sample)
    4: 2.0,   # Kůra / bark      (4.80%, 3842 px — large, hits randomly)
    5: 5.0,   # Trhlina / crack  (0.83%, aspect 0.18 — elongated thin lines)
}


def get_train_transforms(
    patch_size,
    num_samples,
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_oversample: int = 8,
):
    """Build training transforms with patch-level focus on all minority classes.

    Crop ratios are set for every rare class (not just class 6) so the model
    sees rot / crack / knot patches proportionally more often.

    RandAffined uses mode=("bilinear", "nearest") so label maps are never
    interpolated with blended class values.
    """
    ratios = [1.0] * num_classes
    for cls, r in _MINORITY_CROP_RATIOS.items():
        if cls < num_classes and cls != rare_label_idx:
            ratios[cls] = r
    if 0 <= rare_label_idx < num_classes:
        ratios[rare_label_idx] = float(rare_class_oversample)

    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-1000,
                a_max=500,  # Adjusted to wood density to maximize texture contrast
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
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
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=1),
            RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=2),
            # Added 3D Elastic deformation to simulate organic, asymmetrical wood growth
            Rand3DElasticd(
                keys=("image", "label"),
                mode=("bilinear", "nearest"),
                prob=0.15,
                sigma_range=(5, 6),
                magnitude_range=(30, 100),
                padding_mode="reflection",
            ),
            # mode=("bilinear", "nearest") is critical: bilinear on labels
            # creates blended voxel values that don't map to any valid class.
            RandAffined(
                keys=("image", "label"),
                mode=("bilinear", "nearest"),
                prob=0.25,
                rotate_range=(0.12, 0.12, 0.12),
                shear_range=(0.05, 0.05, 0.05),
                translate_range=(10, 10, 10),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="reflection",      # Changed to reflection to prevent edge streaks
            ),
            # Drops out random image boxes to improve robustness; keep labels
            # untouched to avoid corrupting supervision targets.
            RandCoarseDropoutd(
                keys="image",
                holes=5,
                spatial_size=(16, 16, 16),
                prob=0.15,
                fill_value=0,
            ),
            # Non-linear intensity shifts to help isolate low-density air gaps
            RandHistogramShiftd(keys="image", num_control_points=10, prob=0.15),
            RandGaussianNoised(keys="image", prob=0.15, std=0.015),
            # Conservative edge emphasis can help thin crack boundaries.
            RandGaussianSharpend(
                keys="image",
                sigma1_x=(0.5, 1.0),
                sigma1_y=(0.5, 1.0),
                sigma1_z=(0.5, 1.0),
                prob=0.08,
            ),
            # Pair occasional sharpening with occasional blur to mimic
            # reconstruction-kernel variability and avoid over-sharpen bias.
            RandGaussianSmoothd(
                keys="image",
                sigma_x=(0.3, 0.8),
                sigma_y=(0.3, 0.8),
                sigma_z=(0.3, 0.8),
                prob=0.08,
            ),
            RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.75, 1.4)),
            RandScaleIntensityd(keys="image", factors=0.08, prob=0.12),
            EnsureTyped(keys=("image", "label")),
        ]
    )


def get_val_transforms():
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-1000,
                a_max=500,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=("image", "label")),
        ]
    )
