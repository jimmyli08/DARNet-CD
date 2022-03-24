from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss
from utils.msssim import SSIM

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss_ori(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

def hybrid_loss(predictions, target, weight=[0,2,0.2,0.2,0.2,0.2]):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)
    # ssim = SSIM()

    for i,prediction in enumerate(predictions):

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        # ssimloss = ssim(prediction, target)
        loss += weight[i]*(bce + dice) #- ssimloss

    return loss