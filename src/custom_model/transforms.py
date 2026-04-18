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
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandZoomd,
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
    ratios[rare_label_idx] = float(rare_class_oversample)

    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=-1000,
                a_max=3000,
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
            # mode=("bilinear", "nearest") is critical: bilinear on labels
            # creates blended voxel values that don't map to any valid class.
            RandAffined(
                keys=("image", "label"),
                mode=("bilinear", "nearest"),
                prob=0.35,
                rotate_range=(0.15, 0.15, 0.15),
                shear_range=(0.05, 0.05, 0.05),
                translate_range=(10, 10, 10),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
            # Zoom ±15 %: teaches the model to find small structures (rot 108 px,
            # insect 68 px) at multiple scales without distorting their shape.
            RandZoomd(
                keys=("image", "label"),
                prob=0.15,
                min_zoom=0.85,
                max_zoom=1.15,
                mode=("trilinear", "nearest"),
                padding_mode="edge",
            ),
            RandGaussianNoised(keys="image", prob=0.2, std=0.01),
            RandGaussianSmoothd(keys="image", prob=0.15, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
            RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
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
                a_max=3000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            EnsureTyped(keys=("image", "label")),
        ]
    )
