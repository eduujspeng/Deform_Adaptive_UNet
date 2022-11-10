# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/29 20:37
import numpy as np
import torch

from criterion.metrics import dice_coef_metric, iou_coef_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_train(net, dl_train, optimizer, criterion):
    loss_list = []
    dice_list = []
    iou_list = []
    for index, data in enumerate(dl_train, 0):
        train_images, labels = data
        train_images = train_images.to(device)
        labels = labels.to(device)
        outputs = net(train_images)
        loss = criterion(outputs, labels)

        np_labels = np.copy(labels.data.cpu().numpy())
        np_outputs = np.copy(outputs.data.cpu().numpy())
        np_outputs[np.nonzero(np_outputs < 0.5)] = 0.0
        np_outputs[np.nonzero(np_outputs >= 0.5)] = 1.0

        dice = dice_coef_metric(np_outputs, np_labels)
        iou = iou_coef_metric(np_outputs, np_labels)

        loss_list.append(loss.item())
        dice_list.append(dice)
        iou_list.append(iou)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.array(loss_list).mean(), np.array(dice_list).mean(), np.array(iou_list).mean()


def model_test(net, dl_test, criterion):
    loss_list = []
    dice_list = []
    iou_list = []
    with torch.no_grad():
        for data in dl_test:
            test_images, labels = data
            test_images = test_images.to(device)
            labels = labels.to(device)
            outputs = net(test_images)
            loss = criterion(outputs, labels)

            np_labels = np.copy(labels.data.cpu().numpy())
            np_outputs = np.copy(outputs.data.cpu().numpy())
            np_outputs[np.nonzero(np_outputs < 0.48)] = 0.0
            np_outputs[np.nonzero(np_outputs >= 0.48)] = 1.0

            dice = dice_coef_metric(np_outputs, np_labels)
            iou = iou_coef_metric(np_outputs, np_labels)

            loss_list.append(loss.item())
            dice_list.append(dice)
            iou_list.append(iou)
    return np.array(loss_list).mean(), np.array(dice_list).mean(), np.array(iou_list).mean()
