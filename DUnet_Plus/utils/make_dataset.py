# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/28 21:55
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class make_dataset(Dataset):
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
