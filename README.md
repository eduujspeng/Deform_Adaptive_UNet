# Deform_Adaptive_Unet
## overview
### overall model
![overall model image](/image/1.jpg)
    The model uses improved deformable convolution to adaptively extract features; comprehensive interval jump connections are used to gather feature information at the encoder, and residual connections at the decoder to facilitate the transfer of features for exploitation; residual attention convolution optimises the semantic gap between the encoder and the decoder, suppressing extraneous features and highlighting effective features from channel and spatial attention, and adapting to facilitate the transfer of features for exploitation between the two; in addition to improving accuracy, the model uses multiscale depth supervision to further enhance segmentation of the boundary of the focal region and to reduce the over-segmentation of the non-focal region.

### Introduction to the dataset
| Data set name  | kind | address |
| ------------- | ------------- | ------------- |
| kaggle TCGA-LGG  | Brain tumour segmentation  | [https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  |
| ISBI 2012  | cell wall segmentation  | [https://grand-challenge.org/challenges/](https://grand-challenge.org/challenges/)  |
| ISIC 2018  | Dermatological segmentation  | [https://www.kaggle.com/competitions/siim-isic-melanoma-classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification)  |

### Data Enhancement
Training large neural networks on limited datasets requires special attention to the overfitting problem, and this model enhances the data by using random rotation, random scaling, random elastic deformation, and gamma correction enhancement. The detailed code can be found at[brainMRI_prepare.py](./utils/brainMRI_prepare.py)、[ISBI_prepare.py](./utils/ISBI_prepare.py)、[ISIC_prepare.py](./utils/ISIC_prepare.py)

### experimental training
The experiments were conducted in Pytorch deep learning environment, and all algorithms were trained on a single NVIDIA Corporation GP100GL graphics card. The batch size of the experiments on each dataset is 8; the loss function is a mixture of Focal loss and Dice loss; the loss function is optimised using Adam's algorithm; the learning rate is set to 1e-4, and the training is terminated after 300 iterations.

### Project structure
    ├─criterion             # Loss function and evaluation function
    ├─datasets              # Selected data sets
    │  ├─brain_test
    │  ├─ISBI
    │  └─ISIC_test
    ├─image                 # Experimental Pictures
    ├─log                   # log file
    ├─melt                  # Ablation experimental model
    ├─models                # Other experimental models
    ├─model_train           # model training
    ├─pth                   # weights file
    └─utils                 # Tools Documentation

## Usage
### Project Installation
1. **Step 1: Create the conda environment**
```
conda create -n yolo python==3.8
conda activate yolo # Access to the environment
```

2. **Step 2: Clone the project**
```
git clone https://github.com/eduujspeng/Deform_Adaptive_UNet.git
```

3. **Step 3: Install dependencies**
```
cd Deform_Adaptive_UNet
pip install -r ./requirement.txt -U
```
### Project use
```
# Can refer to example-isic-DAU-Net.ipynb

# Image inspection
detection.py
```

## Experimental results
### Ablation experiment
In order to effectively study the performance of different improved algorithms, the effects of different techniques on the network model are verified by replacing part of the network through ablation experiments
Take DU-Net as the baseline model, improve deformable convolution as model 1; add interval jump connection as model 2 on the basis of the baseline model; replace ordinary encoder and decoder splicing with residual attention convolution splicing on the basis of the baseline model as model 3; add multi-scale supervision on the basis of the baseline model as model 4; the algorithm in this paper is used as model 5. Conducting the ablation experiments on the dataset 1. Considering the small data size of dataset 1, ten-fold crossover is used for the verification of segmentation effect
![数据集1折线对比图](/image/图7.jpg)
![数据集1消融实验对比图](/image/图8.jpg)

### Comparison experiments with other algorithms
In order to further verify the generalisation ability and segmentation accuracy of this paper's model in the field of medical image segmentation, four mainstream algorithms, namely, U-Net, U-Net3+, Attention U-Net, and DU-Net, are compared with this paper's algorithm on three datasets
![数据集1实验对比图](/image/图9.jpg)
![数据集2实验对比图](/image/图10.jpg)
![数据集3实验对比图](/image/图11.jpg)



