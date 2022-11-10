# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/30 12:54
import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

from criterion.metrics import dice_coef_metric
from models.Deform_Unet_Plus import Deform_Unet_Plus

if __name__ == '__main__':
    # model load
    pth_root = './pth/brain_Dunet_plus_best.pth'
    net = Deform_Unet_Plus(3, 1)
    net.load_state_dict(torch.load(pth_root, map_location='cpu'))

    # data prepare
    # img_path = 'dataset/ISIC_test/2_img.png'
    # mask_path = 'dataset/ISIC_test/2_mask.png'
    img_path = 'dataset/brain_test/1_img.png'
    mask_path = 'dataset/brain_test/1_mask.png'

    normal_transforms = A.Compose([
        A.RandomCrop(width=128, height=128, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25)
    ])

    img_np = np.array(Image.open(img_path).convert('RGB'))
    mask_np = np.array(Image.open(mask_path).convert('L'))

    arg = normal_transforms(image=img_np, mask=mask_np)

    img_tensor = T.ToTensor()(arg['image'])
    mask_tensor = T.ToTensor()(arg['mask'])

    print(img_tensor.shape, mask_tensor.shape)

    img_tensor.unsqueeze_(dim=0)
    output_tensor = net(img_tensor)
    output_tensor.squeeze_()
    output_tensor = torch.where(output_tensor > 0.48, 1, 0)
    img_tensor.squeeze_()
    mask_tensor.squeeze_()

    print(img_tensor.shape, mask_tensor.shape, output_tensor.shape)

    img_np = img_tensor.numpy()
    mask_np = mask_tensor.numpy()
    output_np = output_tensor.detach().cpu().numpy()

    dice = dice_coef_metric(mask_np,output_np)
    print('dice is {}'.format(dice))

    plt.figure()
    plt.axis('off')
    plt.subplot(131)
    plt.imshow(img_np.transpose(1, 2, 0))
    plt.subplot(132)
    plt.imshow(mask_np)
    plt.subplot(133)
    plt.imshow(output_np)
    plt.show()