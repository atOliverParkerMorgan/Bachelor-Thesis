import inspect
from importlib import import_module

from monai.networks.nets import SwinUNETR

try:
    from monai.networks.nets import UNETR
except ImportError:
    UNETR = None  # type: ignore[assignment]

try:
    from monai.networks.nets import BasicUNetPlusPlus
except ImportError:
    BasicUNetPlusPlus = None  # type: ignore[assignment]

try:
    from monai.networks.nets import MedNeXt
except ImportError:
    MedNeXt = None  # type: ignore[assignment]


def _resolve_segmamba_class():

    candidates = [
        ("segmamba", "SegMamba"),
        ("segmamba.model", "SegMamba"),
        ("segmamba.networks", "SegMamba"),
        ("segmamba.segmamba", "SegMamba"),
    ]
    for module_name, class_name in candidates:
        try:
            module = import_module(module_name)
        except Exception:
            continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
    return None


def _safe_construct(model_cls, kwargs: dict):

    signature = inspect.signature(model_cls.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in signature}
    return model_cls(**filtered_kwargs)


def _build_swin_unetr(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    dropout_path_rate: float = 0.1,
    feature_size: int = 48,
    use_v2: bool = False,
    use_checkpoint: bool = True,
    pretrained_weights: str | None = None,
):
    kwargs = {
        "in_channels": 1,
        "out_channels": num_classes,
        "feature_size": feature_size,
        "use_checkpoint": use_checkpoint,
        "spatial_dims": 3,
    }

    signature = inspect.signature(SwinUNETR.__init__).parameters

    # MONAI changed these arg names across versions
    if "img_size" in signature:
        kwargs["img_size"] = img_size
    if "drop_path_rate" in signature:
        kwargs["drop_path_rate"] = dropout_path_rate
    if "dropout_path_rate" in signature:
        kwargs["dropout_path_rate"] = dropout_path_rate
    if "use_v2" in signature:
        kwargs["use_v2"] = use_v2

    model = SwinUNETR(**kwargs)

    if pretrained_weights is not None:
        import torch
        weights = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        model.load_from(weights=weights)
        print(f"Loaded pretrained SwinUNETR backbone from: {pretrained_weights}")

    return model


def _build_unetr(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    feature_size: int = 16,
    hidden_size: int = 768,
    mlp_dim: int = 3072,
    num_heads: int = 12,
):
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

    # positional embedding arg was renamed across MONAI versions
    signature = inspect.signature(UNETR.__init__).parameters
    if "pos_embed" in signature:
        kwargs["pos_embed"] = "conv"
    elif "proj_type" in signature:
        kwargs["proj_type"] = "conv"

    return UNETR(**kwargs)


def _build_basic_unetplusplus(
    num_classes: int = 7,
    features: tuple[int, int, int, int, int, int] = (32, 32, 64, 128, 256, 32),
):
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


def _build_mednext(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    feature_size: int = 32,
):
    if MedNeXt is None:
        raise ImportError("MedNeXt is not available in the installed MONAI version.")

    kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": num_classes,
        "num_classes": num_classes,
        "n_classes": num_classes,
        "feature_size": feature_size,
        "n_channels": feature_size,
        "img_size": img_size,
        "patch_size": img_size,
    }
    return _safe_construct(MedNeXt, kwargs)


def _build_segmamba(
    num_classes: int = 7,
    img_size: tuple = (128, 384, 128),
    feature_size: int = 32,
):
    segmamba_cls = _resolve_segmamba_class()
    if segmamba_cls is None:
        raise ImportError("SegMamba is not available. Install a SegMamba package that exports a SegMamba class.")

    kwargs = {
        "spatial_dims": 3,
        "in_channels": 1,
        "input_channels": 1,
        "out_channels": num_classes,
        "num_classes": num_classes,
        "n_classes": num_classes,
        "feature_size": feature_size,
        "embed_dim": feature_size,
        "img_size": img_size,
        "patch_size": img_size,
    }
    return _safe_construct(segmamba_cls, kwargs)


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
    pretrained_weights: str | None = None,
):
    """Return a segmentation model by name.

    Choices: swinunetr, swinunetr_v2, unetr, basicunetplusplus, mednext, segmamba
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
            pretrained_weights=pretrained_weights,
        )
    if name == "swinunetr_v2":
        return _build_swin_unetr(
            num_classes=num_classes,
            img_size=img_size,
            dropout_path_rate=dropout_path_rate,
            feature_size=feature_size,
            use_v2=True,
            use_checkpoint=True,
            pretrained_weights=pretrained_weights,
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
    if name == "mednext":
        return _build_mednext(
            num_classes=num_classes,
            img_size=img_size,
            feature_size=feature_size,
        )
    if name == "segmamba":
        return _build_segmamba(
            num_classes=num_classes,
            img_size=img_size,
            feature_size=feature_size,
        )

    raise ValueError(
        f"Unsupported model_name '{model_name}'. "
        "Use one of: swinunetr, swinunetr_v2, unetr, basicunetplusplus, mednext, segmamba."
    )

