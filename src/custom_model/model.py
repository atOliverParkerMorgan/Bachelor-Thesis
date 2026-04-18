import inspect

from monai.networks.nets import SwinUNETR


def get_swin_model(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    dropout_path_rate: float = 0.1,
):
    """Return a 3-D SwinUNETR segmentation model.

    img_size must match the spatial size of the training patches so that the
    window-based attention partitioning is consistent at training and inference.

    dropout_path_rate applies stochastic depth regularisation to the Swin
    transformer blocks — reduces the train/val Dice gap without changing
    model capacity.
    """
    kwargs = {
        "in_channels": 1,
        "out_channels": num_classes,
        "feature_size": 48,
        "use_checkpoint": True,
    }

    # MONAI API differs by version: some releases require/allow img_size,
    # others removed it from SwinUNETR.__init__.
    if "img_size" in inspect.signature(SwinUNETR.__init__).parameters:
        kwargs["img_size"] = img_size

    if "drop_path_rate" in inspect.signature(SwinUNETR.__init__).parameters:
        kwargs["drop_path_rate"] = dropout_path_rate

    return SwinUNETR(**kwargs)
