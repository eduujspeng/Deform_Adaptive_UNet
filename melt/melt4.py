# -*- coding:utf-8 -*-
# author:peng
# Date：2023/2/23 10:20
import torch
from torch import nn

# DU-Net + 残差注意力卷积
from melt.melt_part import Double_Deform_Block, Double_Normal_Block, conn1, conn2, conn3, conn4, CBAMLayer


class Melt4(nn.Module):
    def __init__(self, input_channels, num_classes, deep=False):
        super(Melt4, self).__init__()

        nb_filter = [16, 32, 64, 128, 256]

        self.deep = deep

        self.pool2 = nn.MaxPool2d(2, 2)
        #         self.pool4 = nn.MaxPool2d(4, 4)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值
        #         self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # scale_factor:放大的倍数  插值

        # encode
        self.conv0_0 = Double_Deform_Block(input_channels, (input_channels + nb_filter[0]) // 2,
                                           nb_filter[0])

        self.conv1_0 = Double_Deform_Block(nb_filter[0], (nb_filter[0] + nb_filter[1]) // 2,
                                           nb_filter[1])

        self.conv2_0 = Double_Normal_Block(nb_filter[1],
                                           (nb_filter[1] + nb_filter[2]) // 2,
                                           nb_filter[2])

        self.conv3_0 = Double_Normal_Block(nb_filter[2],
                                           (nb_filter[2] + nb_filter[3]) // 2,
                                           nb_filter[3])

        self.conv4_0 = Double_Normal_Block(nb_filter[3],
                                           (nb_filter[3] + nb_filter[4]) // 2,
                                           nb_filter[4])

        # connection
        self.connect1 = conn1(nb_filter[3], nb_filter[3])
        self.connect2 = conn2(nb_filter[2], nb_filter[2])
        self.connect3 = conn3(nb_filter[1], nb_filter[1])
        self.connect4 = conn4(nb_filter[0], nb_filter[0])
        self.cbam1 = CBAMLayer(nb_filter[3])
        self.cbam2 = CBAMLayer(nb_filter[2])
        self.cbam3 = CBAMLayer(nb_filter[1])
        self.cbam4 = CBAMLayer(nb_filter[0])

        # decode
        self.conv3_1 = Double_Normal_Block(nb_filter[3] + nb_filter[4],
                                           (nb_filter[3] + nb_filter[4] + nb_filter[3]) // 2,
                                           nb_filter[3])

        self.conv2_2 = Double_Normal_Block(nb_filter[2] + nb_filter[3],
                                           (nb_filter[2] + nb_filter[3] + nb_filter[2]) // 2,
                                           nb_filter[2])

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

        x3_1 = self.conv3_1(torch.cat([self.connect1(self.cbam1(x3_0)), self.up2(x4_0)], 1))

        x2_2 = self.conv2_2(torch.cat([self.connect2(self.cbam2(x2_0)), self.up2(x3_1)], 1))

        x1_3 = self.conv1_3(torch.cat([self.connect3(self.cbam3(x1_0)), self.up2(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([self.connect4(self.cbam4(x0_0)), self.up2(x1_3)], 1))

        output = self.final(x0_4)
        return torch.sigmoid(output)


if __name__ == '__main__':
    inputs = torch.ones((3, 3, 64, 64))
    melt = Melt4(3, 1)
    outputs = melt(inputs)
    print(outputs.shape)
