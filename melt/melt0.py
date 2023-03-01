# -*- coding:utf-8 -*-
# author:peng
# Date：2023/2/23 10:19
import torch
from torch import nn

# U-Net网络
from melt.melt_part import Double_Normal_Block


class Melt0(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(Melt0, self).__init__()
        nb_filter = [32, 64, 128, 256, 512]

        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv1 = Double_Normal_Block(in_channels, (in_channels + nb_filter[0]) // 2, nb_filter[0])
        self.conv2 = Double_Normal_Block(nb_filter[0], (nb_filter[1] + nb_filter[0]) // 2, nb_filter[1])
        self.conv3 = Double_Normal_Block(nb_filter[1], (nb_filter[1] + nb_filter[2]) // 2, nb_filter[2])
        self.conv4 = Double_Normal_Block(nb_filter[2], (nb_filter[2] + nb_filter[3]) // 2, nb_filter[3])
        self.conv5 = Double_Normal_Block(nb_filter[3], (nb_filter[3] + nb_filter[4]) // 2, nb_filter[4])

        self.conv6 = Double_Normal_Block(nb_filter[4] + nb_filter[3], (nb_filter[4] + nb_filter[3] + nb_filter[3]) // 2,
                                         nb_filter[3])
        self.conv7 = Double_Normal_Block(nb_filter[3] + nb_filter[2], (nb_filter[3] + nb_filter[2] + nb_filter[2]) // 2,
                                         nb_filter[2])
        self.conv8 = Double_Normal_Block(nb_filter[2] + nb_filter[1], (nb_filter[2] + nb_filter[1] + nb_filter[1]) // 2,
                                         nb_filter[1])
        self.conv9 = Double_Normal_Block(nb_filter[1] + nb_filter[0], (nb_filter[1] + nb_filter[0] + nb_filter[0]) // 2,
                                         nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        x5 = self.conv5(self.pool(x4))

        x = self.conv6(torch.cat([x4, self.up(x5)], dim=1))
        x = self.conv7(torch.cat([x3, self.up(x)], dim=1))
        x = self.conv8(torch.cat([x2, self.up(x)], dim=1))
        x = self.conv9(torch.cat([x1, self.up(x)], dim=1))
        x = self.final(x)
        return torch.sigmoid(x)