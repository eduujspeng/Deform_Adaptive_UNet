# -*- coding:utf-8 -*-
# author:peng
# Date：2023/2/2 14:52
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

melt0 = pd.read_csv('../log/基线模型.csv')
melt1 = pd.read_csv('../log/模型1.csv')
melt2 = pd.read_csv('../log/模型2.csv')
melt3 = pd.read_csv('../log/模型3.csv')
melt4 = pd.read_csv('../log/模型4.csv')
melt5 = pd.read_csv('../log/模型5.csv')

melt0_train_loss = melt0.loc[:, 'train loss']
melt0_test_loss = melt0.loc[:, 'test loss']
melt0_train_dice = melt0.loc[:, 'train dice']
melt0_test_dice = melt0.loc[:, 'test dice']

melt1_train_loss = melt1.loc[:, 'train loss']
melt1_test_loss = melt1.loc[:, 'test loss']
melt1_train_dice = melt1.loc[:, 'train dice']
melt1_test_dice = melt1.loc[:, 'test dice']

melt2_train_loss = melt2.loc[:, 'train loss']
melt2_test_loss = melt2.loc[:, 'test loss']
melt2_train_dice = melt2.loc[:, 'train dice']
melt2_test_dice = melt2.loc[:, 'test dice']

melt3_train_loss = melt3.loc[:, 'train loss']
melt3_test_loss = melt3.loc[:, 'test loss']
melt3_train_dice = melt3.loc[:, 'train dice']
melt3_test_dice = melt3.loc[:, 'test dice']

melt4_train_loss = melt4.loc[:, 'train loss']
melt4_test_loss = melt4.loc[:, 'test loss']
melt4_train_dice = melt4.loc[:, 'train dice']
melt4_test_dice = melt4.loc[:, 'test dice']

melt5_train_loss = melt5.loc[:, 'train loss']
melt5_test_loss = melt5.loc[:, 'test loss']
melt5_train_dice = melt5.loc[:, 'train dice']
melt5_test_dice = melt5.loc[:, 'test dice']

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'


def plot_model_history(model_name, train_loss, val_loss,
                       train_history, val_history, test_history, test_loss
                       , label, num_epochs):
    x = np.arange(num_epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x, train_loss, label='基线模型', lw=3, c="springgreen")
    plt.plot(x, val_loss, label='模型1', lw=3)

    plt.plot(x, train_history, label='模型2', lw=3)
    plt.plot(x, val_history, label='模型3', lw=3)
    plt.plot(x, test_history, label='模型4', lw=3)
    plt.plot(x, test_loss, label='模型5', lw=3)

    plt.title(f"{model_name}", fontsize=15)
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel(label, fontsize=15)

    plt.show()


plot_model_history('train loss', melt0_train_loss, melt1_train_loss, melt2_train_loss, melt3_train_loss,
                   melt4_train_loss, melt5_train_loss, 'loss', 300)

plot_model_history('test loss', melt0_test_loss, melt1_test_loss, melt2_test_loss, melt3_test_loss,
                   melt4_test_loss, melt5_test_loss, 'loss', 300)

plot_model_history('train dice', melt0_train_dice, melt1_train_dice, melt2_train_dice, melt3_train_dice,
                   melt4_train_dice, melt5_train_dice, 'dice', 300)

plot_model_history('test dice', melt0_test_dice, melt1_test_dice, melt2_test_dice, melt3_test_dice,
                   melt4_test_dice, melt5_test_dice, 'dice', 300)