import inspect

from monai.networks.nets import SwinUNETR


def get_swin_model(num_classes: int = 7, img_size: tuple = (128, 384, 128)):
    """Return a 3-D SwinUNETR segmentation model.

    img_size must match the spatial size of the training patches so that the
    window-based attention partitioning is consistent at training and inference.
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

    return SwinUNETR(**kwargs)
