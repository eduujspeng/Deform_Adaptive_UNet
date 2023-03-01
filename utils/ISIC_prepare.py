# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/28 22:04
import glob
import os
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.make_dataset import make_dataset


def data_preparation(root):
    # root = '../input/isic2018resize'

    input_data = os.path.join(root, 'ISIC2018_inputs')
    inputs_paths = glob.glob(input_data + '/*')

    target_data = os.path.join(root, 'ISIC2018_targets')
    targets_paths = glob.glob(target_data + '/*')

    return sorted(inputs_paths), sorted(targets_paths)


strong_transforms = A.Compose([
    A.Resize(width=256 // 2, height=192 // 2, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #     A.RandomRotate90(p=0.5),
    #     A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),

    # Pixels
    A.RandomBrightnessContrast(p=0.2),
    A.RandomGamma(p=0.25),
    # A.IAAEmboss(p=0.25),
    A.Blur(p=0.05, blur_limit=3),
    A.CLAHE(p=0.15),  # 自适应直方图均衡
    A.Sharpen(p=0.15),  # 锐化输入图像并将结果与原始图像叠加
    #     A.ToGray(p=0.3),           # 将输入的 RGB 图像转换为灰度

    A.OneOf([
        A.ChannelShuffle(p=0.15),  # 随机重新排列输入 RGB 图像的通道
        A.ColorJitter(p=0.45),  # 随机改变图像的亮度、对比度和饱和度
        A.FancyPCA(p=0.25),  # 使用FancyPCA增强RGB图像
        A.HueSaturationValue(p=0.15),  # 随机改变输入图像的色调、饱和度和值
    ], p=1.0),
])

normal_transforms = A.Compose([
    A.Resize(width=256 // 2, height=192 // 2, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #     A.RandomRotate90(p=0.5),
    #     A.Transpose(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
])

if __name__ == '__main__':
    # metric
    img, mask = data_preparation(r'D:\data\ISIC')
    train_img,val_img,train_mask,val_mask = train_test_split(img, mask, test_size=0.1, shuffle=True, random_state=42)  # random_state：可以接收int，随机种子实例，或者None。
    train_img,test_img,train_mask,test_mask = train_test_split(train_img,train_mask,test_size=0.1,shuffle=True)

    train_data = make_dataset(train_img,train_mask,strong_transforms)
    val_data = make_dataset(val_img,val_mask,normal_transforms)
    test_data = make_dataset(test_img,test_mask,normal_transforms)

    train_dl = DataLoader(train_data,batch_size=2,shuffle=True)
    val_dl = DataLoader(val_data,batch_size=2)
    test_dl = DataLoader(test_data,batch_size=2)

    print(len(train_dl),len(val_dl),print(test_dl))

    img, label = next(iter(train_dl))
    print(img.shape, label.shape)
    from matplotlib import pyplot as plt

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(zip(img[:2], label[:2])):
        img, label = data
        img = img.numpy().transpose(1, 2, 0)
        label = label.squeeze().numpy()
        print(img.max(), img.min())
        print(label.max(), label.min())
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.subplot(2, 4, i + 5)
        plt.imshow(label, cmap='gray')
    plt.show()

