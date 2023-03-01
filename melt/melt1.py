# -*- coding:utf-8 -*-
# author:peng
# Date：2023/2/23 10:19
import torch
from torch import nn


# DU-Net
from melt.melt_part import Double_Deform_Block, Double_Normal_Block


class Melt1(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Melt1, self).__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.pool2 = nn.MaxPool2d(2, 2)
        #         self.pool4 = nn.MaxPool2d(4, 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        #         self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值

        # encode
        self.conv0_0 = Double_Deform_Block(input_channels, (input_channels + nb_filter[0]) // 2, nb_filter[0])

        self.conv1_0 = Double_Deform_Block(nb_filter[0], (nb_filter[0] + nb_filter[1]) // 2, nb_filter[1])

        self.conv2_0 = Double_Normal_Block(nb_filter[1], (nb_filter[1] + nb_filter[2]) // 2, nb_filter[2])

        self.conv3_0 = Double_Normal_Block(nb_filter[2], (nb_filter[2] + nb_filter[3]) // 2, nb_filter[3])

        self.conv4_0 = Double_Normal_Block(nb_filter[3], (nb_filter[3] + nb_filter[4]) // 2, nb_filter[4])

        # decode
        self.conv3_1 = Double_Normal_Block(nb_filter[3] + nb_filter[4],
                                           (nb_filter[3] + nb_filter[4] + nb_filter[3]) // 2,
                                           nb_filter[3])

        self.conv2_2 = Double_Normal_Block(nb_filter[2] + nb_filter[3],
                                           (nb_filter[2] + nb_filter[3] + nb_filter[
                                               2]) // 2, nb_filter[2])

        self.conv1_3 = Double_Deform_Block(nb_filter[1] + nb_filter[2],
                                           (nb_filter[1] + nb_filter[2] + nb_filter[1]) // 2,
                                           nb_filter[1])

        self.conv0_4 = Double_Deform_Block(nb_filter[0] + nb_filter[1],
                                           (nb_filter[0] + nb_filter[1] + nb_filter[0]) // 2,
                                           nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool2(x0_0))

        x2_0 = self.conv2_0(self.pool2(x1_0))

        x3_0 = self.conv3_0(self.pool2(x2_0))

        x4_0 = self.conv4_0(self.pool2(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up2(x4_0)], 1))

        x2_2 = self.conv2_2(torch.cat([x2_0, self.up2(x3_1)], 1))

        x1_3 = self.conv1_3(torch.cat([x1_0, self.up2(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, self.up2(x1_3)], 1))

        output = self.final(x0_4)
        return torch.sigmoid(output)
