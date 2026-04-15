import torch
import inspect
from monai.losses import DiceCELoss


def get_loss(
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_weight: float = 30.0,
) -> DiceCELoss:
    """Return a DiceCE loss with elevated CE weight for the rare class.

    Class ``rare_label_idx`` (Poškození hmyzem = 6) receives a CE weight of
    ``rare_class_weight`` (30×) so mis-classifying its voxels costs
    proportionally more — matching the strategy in
    nnUNetTrainerRareClassBoostWandb.
    """
    ce_weights = torch.ones(num_classes)
    ce_weights[rare_label_idx] = rare_class_weight

    kwargs = dict(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )

    # MONAI changed CE-weight keyword names across versions. Pick the one
    # supported by the installed DiceCELoss to avoid runtime crashes.
    param_names = inspect.signature(DiceCELoss.__init__).parameters
    if "ce_weight" in param_names:
        kwargs["ce_weight"] = ce_weights
    elif "ce_weights" in param_names:
        kwargs["ce_weights"] = ce_weights
    elif "weight" in param_names:
        kwargs["weight"] = ce_weights

    return DiceCELoss(**kwargs)
