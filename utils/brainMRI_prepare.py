# -*- coding:utf-8 -*-
# author:peng
# Date：2022/9/9 19:34
import glob
import os

import albumentations as A
import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.make_dataset import make_dataset

PATCH_SIZE = 128  # 256


def data_transforms():
    strong_transforms = A.Compose([
        A.RandomResizedCrop(width=PATCH_SIZE, height=PATCH_SIZE, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),

        # Pixels
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.25),
        # A.IAAEmboss(p=0.25),
        A.Blur(p=0.01, blur_limit=3),

        # Affine
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8),

        # A.Normalize(p=1.0), # https://albumentations.readthedocs.io/en/latest/api/pytorch.html?highlight=ToTensor
        # #albumentations.pytorch.transforms.ToTensor ToTensor(),
    ])

    normal_transforms = A.Compose([
        A.RandomCrop(width=PATCH_SIZE, height=PATCH_SIZE, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25)
    ])
    return strong_transforms, normal_transforms


def readfile(file_path):
    dirs = glob.glob(file_path + r'\*')
    # print(dirs)

    data_img = []
    data_label = []
    for subdir in dirs:
        if 'TCGA' in subdir:
            for filename in os.listdir(subdir):
                img_path = subdir + '/' + filename
                # print(img_path)
                if 'mask' in img_path:
                    data_label.append(img_path)
                else:
                    data_img.append(img_path)

    print(len(data_img), len(data_label))

    data_newimg = []
    data_newlabel = []
    for i in data_label:
        value = cv2.imread(i)
        value = value.max()
        if value > 0:
            data_newlabel.append(i)
            i_img = i[:-9] + '.tif'
            data_newimg.append(i_img)
    print(len(data_newimg), len(data_newlabel))
    return data_newimg, data_newlabel


if __name__ == '__main__':
    data_newimg, data_newlabel = readfile(r'D:\data\lgg-mri-segmentation\kaggle_3m')  # 路径不能出现中文
    strong_transforms,normal_transforms = data_transforms()

    print(len(data_newimg), len(data_newlabel))
    data = make_dataset(data_newimg, data_newlabel, strong_transforms)
    train_data, val_data = train_test_split(data, train_size=0.1)
    print(len(train_data))
    train_data, test_data = train_test_split(train_data, test_size=0.1)
    print(len(train_data))
    dl_train = DataLoader(train_data, batch_size=2, shuffle=True)
    dl_val = DataLoader(val_data, batch_size=2)
    dl_test = DataLoader(test_data, batch_size=2)

    print(len(dl_train), len(dl_test), len(dl_val))

    img, label = next(iter(dl_train))
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
