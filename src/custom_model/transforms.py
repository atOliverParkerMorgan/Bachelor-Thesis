from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandAdjustContrastd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    SpatialPadd,
)


def get_train_transforms(patch_size, num_samples):
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS", labels=None),
            ScaleIntensityRanged(
                keys="image",
                a_min=-1000,
                a_max=3000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            SpatialPadd(keys=("image", "label"), spatial_size=patch_size),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
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
            Orientationd(keys=("image", "label"), axcodes="RAS", labels=None),
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
