from monai.networks.nets import UNet, SwinUNETR


def get_model(num_classes=7):
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        channels=(32, 64, 128, 256, 320),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    )

# Option B: SwinUNETR with pretrained weights (best for small datasets)
def get_swin_model(num_classes=7):
    return SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=1,
        out_channels=num_classes,
        feature_size=48,
        use_checkpoint=True,
    )
