import inspect

from monai.networks.nets import SwinUNETR

try:
    from monai.networks.nets import UNETR
except ImportError:
    UNETR = None  # type: ignore[assignment]

try:
    from monai.networks.nets import BasicUNetPlusPlus
except ImportError:
    BasicUNetPlusPlus = None  # type: ignore[assignment]


def _build_swin_unetr(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    dropout_path_rate: float = 0.1,
    feature_size: int = 48,
    use_v2: bool = False,
    use_checkpoint: bool = True,
):
    """Build a 3D SwinUNETR model with MONAI-version-safe kwargs."""

    kwargs = {
        "in_channels": 1,
        "out_channels": num_classes,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
        "spatial_dims": 3,
    }

    signature = inspect.signature(SwinUNETR.__init__).parameters

    # MONAI API differs by version: some releases require/allow img_size,
    # others removed it from SwinUNETR.__init__.
    if "img_size" in signature:
        kwargs["img_size"] = img_size

    # Support both historical and current arg names.
    if "drop_path_rate" in signature:
        kwargs["drop_path_rate"] = dropout_path_rate
    if "dropout_path_rate" in signature:
        kwargs["dropout_path_rate"] = dropout_path_rate

    if "use_v2" in signature:
        kwargs["use_v2"] = use_v2

    return SwinUNETR(**kwargs)


def _build_unetr(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
):
    """Build a 3D UNETR model with MONAI-version-safe kwargs."""

    if UNETR is None:
        raise ImportError("UNETR is not available in the installed MONAI version.")

    if hidden_size % num_heads != 0:
        raise ValueError(
            f"UNETR hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})."
        )

    kwargs = {
        "in_channels": 1,
        "out_channels": num_classes,
        "img_size": img_size,
        "feature_size": feature_size,
        "hidden_size": hidden_size,
        "mlp_dim": mlp_dim,
        "num_heads": num_heads,
        "spatial_dims": 3,
    }

    signature = inspect.signature(UNETR.__init__).parameters
    # Handle older/newer naming for positional embedding argument.
    if "pos_embed" in signature:
        kwargs["pos_embed"] = "conv"
    elif "proj_type" in signature:
        kwargs["proj_type"] = "conv"

    return UNETR(**kwargs)


def _build_basic_unetplusplus(
    num_classes: int = 7,
    features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32),
):
    """Build a 3D BasicUNetPlusPlus CNN baseline."""

    if BasicUNetPlusPlus is None:
        raise ImportError("BasicUNetPlusPlus is not available in the installed MONAI version.")

    if len(features) != 6:
        raise ValueError("BasicUNetPlusPlus requires exactly 6 feature values.")

    return BasicUNetPlusPlus(
        spatial_dims=3,
        in_channels=1,
        out_channels=num_classes,
        features=features,
        deep_supervision=False,
    )


def get_model(
    model_name: str = "swinunetr",
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    dropout_path_rate: float = 0.1,
    feature_size: int = 48,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
    basicunet_features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32),
):
    """Return a segmentation model selected by name.

    Supported models:
      - ``swinunetr``: SwinUNETR (v1)
      - ``swinunetr_v2``: SwinUNETR (v2, residual block at each stage)
      - ``unetr``: UNETR
      - ``basicunetplusplus``: BasicUNetPlusPlus
    """

    name = model_name.lower()
    if name == "swinunetr":
        return _build_swin_unetr(
            num_classes=num_classes,
            img_size=img_size,
            dropout_path_rate=dropout_path_rate,
            feature_size=feature_size,
            use_v2=False,
            use_checkpoint=True,
        )
    if name == "swinunetr_v2":
        return _build_swin_unetr(
            num_classes=num_classes,
            img_size=img_size,
            dropout_path_rate=dropout_path_rate,
            feature_size=feature_size,
            use_v2=True,
            use_checkpoint=True,
        )
    if name == "unetr":
        return _build_unetr(
            num_classes=num_classes,
            img_size=img_size,
            feature_size=feature_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
        )
    if name == "basicunetplusplus":
        return _build_basic_unetplusplus(
            num_classes=num_classes,
            features=basicunet_features,
        )

    raise ValueError(
        f"Unsupported model_name '{model_name}'. "
        "Use one of: swinunetr, swinunetr_v2, unetr, basicunetplusplus."
    )


def get_swin_model(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    dropout_path_rate: float = 0.1,
):
    """Backward-compatible wrapper for existing callers.

    img_size must match the spatial size of the training patches so that the
    window-based attention partitioning is consistent at training and inference.

    dropout_path_rate applies stochastic depth regularisation to the Swin
    transformer blocks — reduces the train/val Dice gap without changing
    model capacity.
    """
    return get_model(
        model_name="swinunetr",
        num_classes=num_classes,
        img_size=img_size,
        dropout_path_rate=dropout_path_rate,
    )
