# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/29 11:35
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from criterion.losses import Focal_dice_loss
from model_train import model_train, model_test
from models.Deform_Unet_Plus import Deform_Unet_Plus

from utils.brainMRI_prepare import data_transforms, readfile
from utils.make_dataset import make_dataset

# 损失函数，网络模型，数据可以使用GPU运算
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # data prepare
    data_newimg, data_newlabel = readfile(r'D:\data\lgg-mri-segmentation\kaggle_3m')  # 路径不能出现中文
    strong_transforms, normal_transforms = data_transforms()
    # random_state：可以接收int，随机种子实例，或者None。
    train_img, val_img, train_mask, val_mask = train_test_split(data_newimg, data_newlabel, test_size=0.1, shuffle=True,
                                                                random_state=42)

    train_img, test_img, train_mask, test_mask = train_test_split(train_img, train_mask, test_size=0.1,
                                                                  shuffle=True)

    # loss
    criterion = Focal_dice_loss()
    criterion = criterion.to(device)

    # model
    net = Deform_Unet_Plus(3, 1).to(device)

    # optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    train_loss_list = []
    train_dice_list = []
    train_iou_list = []

    test_loss_list = []
    test_dice_list = []
    test_iou_list = []
    tmp = 0.8
    for epoch in range(300):
        print('DUnet Plus第{}轮训练开始'.format(epoch + 1))

        train_data = make_dataset(train_img, train_mask, strong_transforms)
        val_data = make_dataset(val_img, val_mask, normal_transforms)
        train_dl = DataLoader(train_data, batch_size=2, shuffle=True)
        val_dl = DataLoader(val_data, batch_size=2)

        net.train()
        train_loss, train_dice, train_iou = model_train(net, train_dl, optimizer, criterion)
        train_loss_list.append(train_loss)
        train_dice_list.append(train_dice)
        train_iou_list.append(train_iou)

        net.eval()
        val_loss, val_dice, val_iou = model_test(net, val_dl, criterion)
        test_loss_list.append(val_loss)
        test_dice_list.append(val_dice)
        test_iou_list.append(val_iou)
        if val_dice > tmp and epoch > 250:
            tmp = val_dice
            torch.save(net.state_dict(), '../pth/DUnet_plus_best.pth')  # model save
            print('best model save')

        print('训练集loss是{}，训练集dice是{}，测试集dice是{}'.format(train_loss, train_dice, val_dice))

    # model load
    net.load_state_dict(torch.load('../pth/DUnet_plus_best.pth', map_location=device))
    dice, iou = 0, 0

    test_data = make_dataset(test_img, test_mask, normal_transforms)
    test_dl = DataLoader(test_data, batch_size=2)
    for i in range(10):
        test_loss, test_dice, test_iou = model_test(net, test_dl, criterion)
        dice += test_dice
        iou += test_iou
    print('the model dice is {}, iou is {}'.format(dice / 10, iou / 10))
