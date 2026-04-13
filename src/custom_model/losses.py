import torch
from monai.losses import DiceCELoss


def get_loss(
    num_classes: int = 7,
    rare_label_idx: int = 6,
    rare_class_weight: float = 30.0,
) -> DiceCELoss:
    """Return a DiceCE loss with elevated CE weight for the rare class.

    Class ``rare_label_idx`` (poskozeni_hmyzem = 6) receives a CE weight of
    ``rare_class_weight`` (30×) so mis-classifying its voxels costs
    proportionally more — matching the strategy in
    nnUNetTrainerRareClassBoostWandb.
    """
    ce_weights = torch.ones(num_classes)
    ce_weights[rare_label_idx] = rare_class_weight
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=0.5,
        lambda_ce=0.5,
        ce_weight=ce_weights,
    )
