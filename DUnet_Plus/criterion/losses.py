import torch
import torch.nn as nn


# 损失函数

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, logits=False, sampling='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.sampling = sampling

    def forward(self, y_pred, y_true):
        alpha = self.alpha
        alpha_ = (1 - self.alpha)
        if self.logits:
            y_pred = torch.sigmoid(y_pred)

        pt_positive = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        pt_negative = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        pt_positive = torch.clamp(pt_positive, 1e-3, .999)
        pt_negative = torch.clamp(pt_negative, 1e-3, .999)
        pos_ = (1 - pt_positive) ** self.gamma
        neg_ = pt_negative ** self.gamma

        pos_loss = -alpha * pos_ * torch.log(pt_positive)
        neg_loss = -alpha_ * neg_ * torch.log(1 - pt_negative)
        loss = pos_loss + neg_loss

        if self.sampling == "mean":
            return loss.mean()
        elif self.sampling == "sum":
            return loss.sum()
        elif self.sampling is None:
            return loss


def dice_coef_loss(inputs, target):
    smooth = 1e-6
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    inputs = torch.where(inputs >= 0.5, 1, 0)
    intersection1 = 2.0 * ((target * inputs).sum()) + smooth
    union1 = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union) + 1 - (intersection1 / union1)


class Focal_dice_loss(nn.Module):
    def __init__(self):
        super(Focal_dice_loss, self).__init__()

    def forward(self, inputs, target):
        dicescore = dice_coef_loss(inputs, target)
        focusloss = FocalLoss()(inputs, target)
        return dicescore + focusloss


if __name__ == '__main__':
    # loss check
    a = torch.ones((4, 1, 4, 4))
    b = a - 0.1
    dice = Focal_dice_loss()(b, a)
    print(dice)
