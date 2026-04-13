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
    ScaleIntensityRanged,
    SpatialPadd,
)


def get_train_transforms(
    patch_size,
    num_samples,
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_oversample: int = 8,
):
    """Build training transforms with patch-level focus on the rare class.

    ``RandCropByLabelClassesd`` is used instead of ``RandCropByPosNegLabeld``
    so that crops from volumes containing label ``rare_label_idx`` are centred
    on that class with ``rare_class_oversample`` times higher probability than
    any other class — mirroring the patch-forcing strategy of
    nnUNetTrainerRareClassBoostWandb.
    """
    ratios = [1.0] * num_classes
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
            RandAffined(
                keys=("image", "label"),
                prob=0.2,
                rotate_range=(0.1, 0.1, 0.1),
                shear_range=(0.05, 0.05, 0.05),
                translate_range=(10, 10, 10),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
            ),
            RandGaussianNoised(keys="image", prob=0.15, std=0.01),
            RandAdjustContrastd(keys="image", prob=0.15, gamma=(0.7, 1.5)),
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
