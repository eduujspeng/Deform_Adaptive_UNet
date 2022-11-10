# -*- coding:utf-8 -*-
# author:peng
# Date：2022/11/2 11:39
import albumentations as A
import cv2
from matplotlib import pyplot as plt

image = cv2.imread(r'../image/Brain_TCGA_CS_4941_19960909_13.png')
mask = cv2.imread(r'../image/Brain_TCGA_CS_4941_19960909_13_mask.png')
print(type(image))
normal_transforms = A.Compose([
    A.Resize(width=256 // 2, height=256 // 2, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(),
    # 对比度受限直方图均衡
    # （Contrast Limited Adaptive Histogram Equalization）
    A.CLAHE(),
    # 随机旋转 90°
    A.RandomRotate90(),
    # 转置
    A.Transpose(),
    # 随机仿射变换
    # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
    # 模糊
    A.OneOf([
        A.Blur(blur_limit=3),
        # 光学畸变
        A.OpticalDistortion(),
        # 网格畸变
        A.GridDistortion(),
        # 随机改变图片的 HUE、饱和度和值
        A.HueSaturationValue()
    ],p=1.0)
], p=1.0)

strong_transforms = A.Compose([
    A.Resize(width=256 // 2, height=256 // 2, p=1.0),
    A.HorizontalFlip(p=1),
    A.VerticalFlip(p=1),
    A.RandomRotate90(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=1),

    # Pixels
    A.RandomBrightnessContrast(p=1),
    A.RandomGamma(p=1),
    # A.IAAEmboss(p=0.25),
    A.Blur(p=1, blur_limit=3),
    A.CLAHE(p=1),  # 自适应直方图均衡
    A.Sharpen(p=1),  # 锐化输入图像并将结果与原始图像叠加
    #     A.ToGray(p=0.3),           # 将输入的 RGB 图像转换为灰度

    A.OneOf([
        A.ChannelShuffle(p=0.15),  # 随机重新排列输入 RGB 图像的通道
        A.ColorJitter(p=0.45),  # 随机改变图像的亮度、对比度和饱和度
        A.FancyPCA(p=0.25),  # 使用FancyPCA增强RGB图像
        A.HueSaturationValue(p=0.15),  # 随机改变输入图像的色调、饱和度和值
    ], p=1.0),
])
RESIZE_SIZE = 128  # or 256
train_transform = A.Compose([
    A.Resize(RESIZE_SIZE, RESIZE_SIZE),
    A.OneOf([
        A.RandomGamma(gamma_limit=(60, 120), p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
    ]),
    A.OneOf([
        A.Blur(blur_limit=4, p=1),
        A.MotionBlur(blur_limit=4, p=1),
        A.MedianBlur(p=1)
    ], p=0.5),
    A.HorizontalFlip(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=1),
    # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
])

image_ = normal_transforms(image=image)['image']
image1_ = strong_transforms(image=image)['image']
image2_ = A.ToGray(p=1.0)(image=image)['image']

print(image2_.shape)


def show(image1, image2):
    plt.figure()
    plt.subplot(121)
    plt.imshow(image1)
    plt.subplot(122)
    plt.imshow(image2)
    plt.savefig('../tmp/1.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()

import torchvision.transforms as T
arg = normal_transforms(image=image,mask=mask)
image = arg['image']
mask = arg['mask']
img_tensor = T.ToTensor()(image)
print(type(image),type(img_tensor))
show(image,mask)

# plt.figure(figsize=(10, 10))
# plt.subplot(131)
# plt.imshow(image_)
#
# plt.subplot(132)
# plt.imshow(image1_)
#
# plt.subplot(133)
# plt.imshow(image2_)
# plt.show()
