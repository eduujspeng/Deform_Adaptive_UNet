# -*- coding:utf-8 -*-
# author:peng
# Date：2022/11/3 9:48
import glob
import albumentations as A
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from criterion.metrics import dice_coef_metric, iou_coef_metric
from models.Deform_Unet_Plus import Deform_Unet_Plus


def test_data_prepare(root):
    pathes = glob.glob(root + '/*')
    image = []
    mask = []
    for path in pathes:
        if 'img' in path:
            image.append(path)
        elif 'mask' in path:
            mask.append(path)
    return sorted(image), sorted(mask)


class ISICDataset(Dataset):
    def __init__(self, img, label, transform):
        self.img = img
        self.label = label
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]

        img_open = Image.open(img).convert('RGB')
        img_open = np.array(img_open)
        #         img_tensor = self.transform(img_open)

        label_open = Image.open(label).convert('L')
        label_open = np.array(label_open)
        #         label_tensor = self.transform(label_open)
        arg = self.transform(image=img_open, mask=label_open)

        return self.to_tensor(arg['image']), self.to_tensor(arg['mask'])

    def __len__(self):
        return len(self.img)



def test(net, dl_test):
    dice_list = []
    iou_list = []
    with torch.no_grad():
        for data in dl_test:
            test_images, labels = data
            test_images = test_images.to(device)
            labels = labels.to(device)
            outputs = net(test_images)

            np_labels = np.copy(labels.data.cpu().numpy())
            np_outputs = np.copy(outputs.data.cpu().numpy())
            np_outputs[np.nonzero(np_outputs < 0.48)] = 0.0
            np_outputs[np.nonzero(np_outputs >= 0.48)] = 1.0

            dice = dice_coef_metric(np_outputs, np_labels)
            iou = iou_coef_metric(np_outputs, np_labels)

            dice_list.append(dice)
            iou_list.append(iou)
            print(dice,iou)
    return np.array(dice_list).mean(), np.array(iou_list).mean()


if True:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A_transform = A.Resize(128, 128)
    img,mask = test_data_prepare(r'./dataset/brain_test')
    data = ISICDataset(img,mask,A_transform)
    test_dl = DataLoader(data,batch_size=4)
    print(len(test_dl))

    net = Deform_Unet_Plus(3,1).to(device)
    net.load_state_dict(torch.load(r'./pth/brain_DUnet_plus_best.pth',map_location=device))

    dice,iou = test(net,test_dl)
    print('dice is {},iou is {}'.format(dice,iou))


