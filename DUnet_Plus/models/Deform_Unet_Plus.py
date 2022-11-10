# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/28 19:36
import torch
from torch import nn

from models.Deform_Unet_Plus_part import CBAMLayer, Double_Deform_Block, Double_Normal_Block, conn1, conn2, conn3, conn4


class Deform_Unet_Plus(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(Deform_Unet_Plus, self).__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(4, 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值

        # encode
        self.conv0_0 = Double_Deform_Block(input_channels, (input_channels + nb_filter[0]) // 2, nb_filter[0])

        self.conv1_0 = Double_Deform_Block(nb_filter[0], (nb_filter[0] + nb_filter[1]) // 2, nb_filter[1])

        self.conv2_0 = Double_Normal_Block(nb_filter[1] + nb_filter[1],
                                           (nb_filter[1] + nb_filter[1] + nb_filter[2]) // 2, nb_filter[2])

        self.conv3_0 = Double_Normal_Block(nb_filter[2] + nb_filter[1],
                                           (nb_filter[1] + nb_filter[2] + nb_filter[3]) // 2, nb_filter[3])

        self.conv4_0 = Double_Normal_Block(nb_filter[3] + nb_filter[1],
                                           (nb_filter[1] + nb_filter[3] + nb_filter[4]) // 2, nb_filter[4])

        self.down_dense1 = nn.Conv2d(nb_filter[0], nb_filter[1], kernel_size=1)
        self.down_dense2 = nn.Conv2d(nb_filter[1], nb_filter[1], kernel_size=1)
        self.down_dense3 = nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1)

        # connection
        self.connect1 = conn1(nb_filter[3], nb_filter[3])
        self.connect2 = conn2(nb_filter[2], nb_filter[2])
        self.connect3 = conn3(nb_filter[1], nb_filter[1])
        self.connect4 = conn4(nb_filter[0], nb_filter[0])

        # decode
        self.conv3_1 = Double_Normal_Block(nb_filter[3] + nb_filter[4],
                                           (nb_filter[3] + nb_filter[4] + nb_filter[3]) // 2,
                                           nb_filter[3])

        self.conv2_2 = Double_Normal_Block(nb_filter[2] + nb_filter[3],
                                           (nb_filter[2] + nb_filter[3]+ nb_filter[2]) // 2, nb_filter[2])

        self.conv1_3 = Double_Deform_Block(nb_filter[1] + nb_filter[2],
                                           (nb_filter[1] + nb_filter[2] + nb_filter[1]) // 2,
                                           nb_filter[1])

        self.conv0_4 = Double_Deform_Block(nb_filter[0] + nb_filter[1],
                                           (nb_filter[0] + nb_filter[1] + nb_filter[0]) // 2,
                                           nb_filter[0])

        self.up_dense1 = nn.Conv2d(nb_filter[4], nb_filter[3], kernel_size=1)
        self.up_dense2 = nn.Conv2d(nb_filter[3], nb_filter[2], kernel_size=1)
        self.up_dense3 = nn.Conv2d(nb_filter[2], nb_filter[1], kernel_size=1)

        self.cbam3_1 = CBAMLayer(128)
        self.cbam2_2 = CBAMLayer(64)
        self.cbam1_3 = CBAMLayer(32)
        self.cbam0_4 = CBAMLayer(16)

        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)

        x1_0 = self.conv1_0(self.pool2(x0_0))

        x2_0 = self.conv2_0(torch.cat([self.down_dense1(self.pool4(x0_0)), self.pool2(x1_0)], dim=1))

        x3_0 = self.conv3_0(torch.cat([self.down_dense2(self.pool4(x1_0)), self.pool2(x2_0)], dim=1))

        x4_0 = self.conv4_0(torch.cat([self.down_dense3(self.pool4(x2_0)), self.pool2(x3_0)], dim=1))

        x3_1 = self.conv3_1(torch.cat([self.cbam3_1(self.connect1(x3_0)), self.up2(x4_0)], 1))

        tmp = self.up_dense1(self.up4(x4_0))
        x2_2 = self.conv2_2(
            torch.cat([self.cbam2_2(self.connect2(x2_0)), self.up2(x3_1) + self.up_dense1(self.up4(x4_0))], 1))

        tmp = self.up_dense2(self.up4(x3_1))
        x1_3 = self.conv1_3(
            torch.cat([self.cbam1_3(self.connect3(x1_0)), self.up2(x2_2) + self.up_dense2(self.up4(x3_1))], 1))

        tmp = self.up_dense3(self.up4(x2_2))
        x0_4 = self.conv0_4(
            torch.cat([self.cbam0_4(self.connect4(x0_0)), self.up2(x1_3) + self.up_dense3(self.up4(x2_2))], 1))

        output = self.final(x0_4)
        return torch.sigmoid(output)


if __name__ == '__main__':
    net = Deform_Unet_Plus(input_channels=3, num_classes=1)
    inputs = torch.ones((2, 3, 64, 64))
    print(net(inputs).shape)