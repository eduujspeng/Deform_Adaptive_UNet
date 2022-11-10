# -*- coding:utf-8 -*-
# author:peng
# Date：2022/10/28 19:36
import torch
import torch.nn as nn

from torchvision.ops import DeformConv2d
# from models.deform_conv_v2 import DeformConv2d


class connection(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(connection, self).__init__()

        self.conv3_3 = nn.Conv2d(in_ch, ou_ch // 2, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv2d(in_ch, ou_ch // 2, kernel_size=1)

    def forward(self, x):
        x1 = self.conv3_3(x)
        x2 = self.conv1_1(x)
        x = torch.cat([x1, x2], dim=1)
        return x


class conn1(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(conn1, self).__init__()

        self.conn = connection(in_ch, ou_ch)
        self.conn1 = connection(ou_ch, ou_ch)

    def forward(self, x):
        x1 = self.conn(x)
        return self.conn1(x1)


class conn2(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(conn2, self).__init__()

        self.conn = connection(in_ch, ou_ch)
        self.conn1 = connection(ou_ch, ou_ch)

        self.conn2 = connection(ou_ch, ou_ch)
        self.conn3 = connection(ou_ch, ou_ch)

    def forward(self, x):
        x1 = self.conn(x)
        x2 = self.conn1(x1)
        x3 = self.conn2(x2)
        return self.conn3(x3)


class conn3(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(conn3, self).__init__()
        self.conn = connection(in_ch, ou_ch)
        self.conn1 = connection(ou_ch, ou_ch)

        self.conn2 = connection(ou_ch, ou_ch)
        self.conn3 = connection(ou_ch, ou_ch)

        self.conn4 = connection(ou_ch, ou_ch)
        self.conn5 = connection(ou_ch, ou_ch)

    def forward(self, x):
        x1 = self.conn(x)
        x2 = self.conn1(x1)
        x3 = self.conn2(x2)
        x4 = self.conn3(x3)
        x5 = self.conn4(x4)
        return self.conn5(x5)


class conn4(nn.Module):
    def __init__(self, in_ch, ou_ch):
        super(conn4, self).__init__()
        self.conn = connection(in_ch, ou_ch)
        self.conn1 = connection(ou_ch, ou_ch)

        self.conn2 = connection(ou_ch, ou_ch)
        self.conn3 = connection(ou_ch, ou_ch)

        self.conn4 = connection(ou_ch, ou_ch)
        self.conn5 = connection(ou_ch, ou_ch)

        self.conn6 = connection(ou_ch, ou_ch)
        self.conn7 = connection(ou_ch, ou_ch)

    def forward(self, x):
        x1 = self.conn(x)
        x2 = self.conn1(x1)
        x3 = self.conn2(x2)
        x4 = self.conn3(x3)
        x5 = self.conn4(x4)
        x6 = self.conn5(x5)
        x7 = self.conn6(x6)
        return self.conn7(x7)


class Double_Deform_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1_ = nn.Conv2d(in_channels, 2 * 3 * 3, kernel_size=3, stride=1, padding=1)
        self.conv1 = DeformConv2d(in_channels, middle_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2_ = nn.Conv2d(middle_channels, 2 * 3 * 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = DeformConv2d(middle_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        offset1 = self.conv1_(x)
        out = self.conv1(x, offset1)
        out = self.bn1(out)
        out = self.relu(out)

        offset2 = self.conv2_(out)
        out = self.conv2(out, offset2)
        out = self.bn2(out)
        out = self.relu(out)

        return out

# class Double_Deform_Block(nn.Module):
#     def __init__(self, in_channels, middle_channels, out_channels):
#         super().__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = DeformConv2d(in_channels, middle_channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(middle_channels)
#         self.conv2 = DeformConv2d(middle_channels, out_channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         return out


class Double_Normal_Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


if __name__ == '__main__':
    deform_block = Double_Deform_Block(in_channels=3, middle_channels=9, out_channels=32)
    a = torch.ones((3, 3, 8, 8))
    print(deform_block(a).shape)
