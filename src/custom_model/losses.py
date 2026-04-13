from monai.losses import DiceCELoss


def get_loss():
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=0.5,
        lambda_ce=0.5,
    )
