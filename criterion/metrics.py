import torch

# 评估指标
def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0
    return intersection / union


def iou_coef_metric(output, target):
    intersection = ((output == 1) & (target == 1)).sum()
    union = ((output == 1) | (target == 1)).sum()

    if target.sum() == 0 and output.sum() == 0:
        return 1.0
    return intersection / union

if __name__ == '__main__':
    # Metric check
    dice = dice_coef_metric(torch.tensor([0., 0.9]),
                     torch.tensor([1., 1.]))
    iou = iou_coef_metric(torch.tensor([1., 1.]),
                    torch.tensor([0., 1.]))
    print(dice,iou)