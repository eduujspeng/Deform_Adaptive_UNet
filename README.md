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
在有限的数据集上训练大型神经网络，需要特别注意过拟合问题，本模型通过使用随机旋转、随机缩放、随机弹性变形、伽马校正增强等方式进行数据增强。详细代码可见[brainMRI_prepare.py](./utils/brainMRI_prepare.py)、[ISBI_prepare.py](./utils/ISBI_prepare.py)、[ISIC_prepare.py](./utils/ISIC_prepare.py)

### 实验训练
实验基于 Pytorch 深度学习环境进行, 所有算法均在单张 NVIDIA Corporation GP100GL显卡上训练完成。每个数据集上实验的批处理大小均为 8；损失函数采用Focal loss和Dice loss的混合组成；使用 Adam 算法对损失函数进行优化；学习率设为1e-4,共迭代 300 次后终止训练。

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

## 使用方法
### 项目安装
1. **第一步：创建conda环境**
```
conda create -n yolo python==3.8
conda activate yolo # 进入环境
```

2. **第二步：克隆项目**
```
git clone https://github.com/eduujspeng/Deform_Adaptive_UNet.git
```

3. **第三步：安装依赖**
```
cd Deform_Adaptive_UNet
pip install -r ./requirement.txt -U
```
### 项目使用
```
# 数据集1训练
BrainMRI_train.py

# 数据集2训练
ISBI_train.py

# 数据集3训练
ISIC_train.py

# 图片检验
detection.py
```

## 实验结果
### 消融实验
为了有效研究不同改进算法的性能，通过消融实验替换部分网络的方式验证不同技术对网络模型的影响
以U-Net为基线模型，可变形卷积替换成普通卷积为模型1；在模型1的基础上添加间隔跳跃连接为模型2；在模型1的基础上将普通编码器、解码器拼接替换成残差注意力卷积拼接为模型3；在模型1的基础上添加多尺度监督为模型4；本文算法作为模型5。在数据集1上进行消融实验，考虑到数据集1的数据规模较小，采用十折交叉的方法进行分割效果的验证
![数据集1折线对比图](/image/折线图对比.png)
![数据集1消融实验对比图](/image/消融实验对比.jpg)

### 其他算法对比实验
为了进一步验证本文模型在医学图像分割领域的泛化能力和分割精度，在三个数据集上将U-Net、U-Net3+、Attention U-Net、DU-Net这四种主流算法与本文算法进行比较
![数据集1实验对比图](/image/数据集1实验对比.jpg)
![数据集2实验对比图](/image/数据集2实验对比.jpg)
![数据集3实验对比图](/image/数据集3实验对比.jpg)



