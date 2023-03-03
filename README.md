# Deform_Adaptive_Unet
## 概述
### 整体模型
![整体模型图](/image/DAU-Net模型.jpg)
    模型使用可变形卷积自适应提取特征；利用全面的间隔跳跃连接在编码器上搜集特征信息，解码器上残差连接促进特征进行传递利用；残差注意力卷积优化编码器和解码器之间的语义差距，从通道和空间上的注意力抑制无关特征，突出有效特征，自适应促进二者之间的特征传递利用；除了提高精度外，模型还使用多尺度深度监督来进一步增强病灶区域边界分割并减少非病灶区域的过度分割。

### 数据集介绍
| 数据集名称  | 种类 | 地址 |
| ------------- | ------------- | ------------- |
| kaggle TCGA-LGG  | 脑肿瘤分割  | [https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  |
| ISBI 2012  | 细胞壁分割  | [https://grand-challenge.org/challenges/](https://grand-challenge.org/challenges/)  |
| ISIC 2018  | 皮肤病分割  | [https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/164560](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/discussion/164560)  |

### 数据增强
在有限的数据集上训练大型神经网络，需要特别注意过拟合问题，本模型通过使用随机旋转、随机缩放、随机弹性变形、伽马校正增强等方式进行数据增强。详细代码可见brainMRI_prepare.py、ISBI_prepare.py、ISIC_prepare.py

### 实验训练
实验基于 Pytorch 深度学习环境进行, 所有算法均在单张 NVIDIA Corporation GP100GL显卡上训练完成。每个数据集上实验的批处理大小均为 8；损失函数采用Focal loss[21]和Dice loss[22]的混合组成；使用 Adam 算法对损失函数进行优化；学习率设为1e-4,共迭代 300 次后终止训练。

### 项目结构
    ├─criterion             # 损失函数和评估函数
    ├─datasets              # 部分数据集
    │  ├─brain_test
    │  ├─ISBI
    │  └─ISIC_test
    ├─image                 # 实验图片
    ├─log                   # 日志文件
    ├─melt                  # 消融实验模型
    ├─models                # 其他实验模型
    ├─model_train           # 模型训练
    ├─pth                   # 权重文件
    └─utils                 # 工具文件
    
