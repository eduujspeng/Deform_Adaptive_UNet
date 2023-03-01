# -*- coding:utf-8 -*-
# author:peng
# Date：2022/9/7 15:40

"""
查看模型参数量
DUnet plus:
=================================================================================================================================================================
Total params: 3,047,837
-----------------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 49.64MB
Total MAdd: 3.42GMAdd
Total Flops: 1.72GFlops
Total MemR+W: 111.45MB

DUnet:
=========================================================================================================================================================
Total params: 9,900,999
---------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 52.56MB
Total MAdd: 7.36GMAdd
Total Flops: 3.68GFlops
Total MemR+W: 120.75MB

Unet:
=======================================================================================================================================================
Total params: 9,807,151
-------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 46.94MB
Total MAdd: 11.23GMAdd
Total Flops: 5.62GFlops
Total MemR+W: 130.16MB

AttentionUnet:
==========================================================================================================================================================
Total params: 8,726,077
----------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 74.31MB
Total MAdd: 8.35GMAdd
Total Flops: 4.18GFlops
Total MemR+W: 179.04MB

Unet3+:
============================================================================================================================================================
Total params: 6,748,993
------------------------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 183.16MB
Total MAdd: 24.79GMAdd
Total Flops: 12.41GFlops
Total MemR+W: 322.9MB
"""
import torch
from torchsummary import summary
from torchstat import stat

from models.DUnet import DUnet
from models.Deform_Adaptive_UNet import Deform_Adaptive_UNet
from models.attiontion_unet import AttU_Net

from models.unet import UNet
from models.unet_3plus import UNet3Plus

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def lookNetParameters(net, root):
    net.load_state_dict(torch.load(root, map_location=device))
    summary(net, input_size=(3, 128, 128), batch_size=-1, device='cpu')


def lookNetParameters1(net, root):
    net.load_state_dict(torch.load(root, map_location=device),strict=False)
    stat(net, (3, 128, 128))


if __name__ == '__main__':
    deform_unet_plus_pth = '../pth/ISIC_Dunet_plus_best.pth'
    deform_unet_plus = Deform_Adaptive_UNet(3, 1,deep = True)
    # lookNetParameters1(deform_unet_plus,deform_unet_plus_pth)

    DUnet_pth = '../pth/ISBI_Dunet_best.pth'
    DUNetV1V2 = DUnet(3, 1)
    # stat(DUNetV1V2,(3,128,128))
    # lookNetParameters1(DUNetV1V2,DUnet_pth)

    Unet_pth = '../pth/ISBI_Unet_best.pth'
    unet = UNet(3, 1)
    # stat(unet, (3, 128, 128))
    # lookNetParameters1(unet,Unet_pth)

    Attention_Unet_pth = '../pth/ISBI_attentionUnet_best.pth'
    Attention_Unet = AttU_Net(3, 1)
    # stat(Attention_Unet,(3,128,128))
    # lookNetParameters1(Attention_Unet, Attention_Unet_pth)

    unet_3plus = UNet3Plus(3,1)
    stat(unet_3plus, (3, 128, 128))


