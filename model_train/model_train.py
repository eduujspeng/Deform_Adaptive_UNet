# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/29 20:37
import numpy as np
import torch

from criterion.metrics import dice_coef_metric, iou_coef_metric

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# deep:是否使用深度监督
def model_train(net, dl_train, optimizer, criterion, deep=False):
    loss_list = []
    dice_list = []
    iou_list = []
    for index, data in enumerate(dl_train, 0):
        train_images, labels = data
        train_images = train_images.to(device)
        labels = labels.to(device)
        outputs = net(train_images)
        # 模型使用深度监督
        loss_sum = torch.tensor(0.).to(device)
        if deep:
            w = 0.
            for output in outputs:
                w += 0.1
                loss = criterion(output, labels)
                loss_sum += (w * loss)
            np_outputs = np.copy(outputs[-1].data.cpu().numpy())
        else:
            loss = criterion(outputs, labels)
            loss_sum += loss
            np_outputs = np.copy(outputs.data.cpu().numpy())

        np_labels = np.copy(labels.data.cpu().numpy())

        np_outputs[np.nonzero(np_outputs < 0.5)] = 0.0
        np_outputs[np.nonzero(np_outputs >= 0.5)] = 1.0

        dice = dice_coef_metric(np_outputs, np_labels)
        iou = iou_coef_metric(np_outputs, np_labels)

        loss_list.append(loss_sum.item())
        dice_list.append(dice)
        iou_list.append(iou)

        optimizer.zero_grad()
        loss_sum.backward()
        optimizer.step()
    return np.array(loss_list).mean(), np.array(dice_list).mean(), np.array(iou_list).mean()


def model_test(net, dl_test, criterion, deep=False):
    loss_list = []
    dice_list = []
    iou_list = []
    with torch.no_grad():
        for data in dl_test:
            test_images, labels = data
            test_images = test_images.to(device)
            labels = labels.to(device)
            outputs = net(test_images)
            if deep:
                outputs = outputs[-1]
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
