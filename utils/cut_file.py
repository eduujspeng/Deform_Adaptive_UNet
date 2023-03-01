# -*- coding:utf-8 -*-
# author:peng
# Date：2023/2/23 19:56
# 使用PIL库进行图片切割
import glob

from PIL import Image

imgs = glob.glob('../dataset/ISBI/train/image/*')
labels = glob.glob('../dataset/ISBI/train/label/*')
imgs = sorted(imgs)
labels = sorted(labels)

index = 0
for img,label in zip(imgs,labels):
    img_pil = Image.open(img)
    label_pil = Image.open(label)

    # 获得图片尺寸
    w, h = img_pil.size

    # 切割成4张小图
    im1 = img_pil.crop((0, 0, w // 2, h // 2))
    im2 = img_pil.crop((w // 2, 0, w, h // 2))
    im3 = img_pil.crop((0, h // 2, w // 2, h))
    im4 = img_pil.crop((w // 2, h // 2, w, h))

    label1 = label_pil.crop((0, 0, w // 2, h // 2))
    label2 = label_pil.crop((w // 2, 0, w, h // 2))
    label3 = label_pil.crop((0, h // 2, w // 2, h))
    label4 = label_pil.crop((w // 2, h // 2, w, h))

    # 保存在本地
    im1.save("../dataset/ISBI/cut/{}_image_part1.png".format(index))
    im2.save("../dataset/ISBI/cut/{}_image_part2.png".format(index))
    im3.save("../dataset/ISBI/cut/{}_image_part3.png".format(index))
    im4.save("../dataset/ISBI/cut/{}_image_part4.png".format(index))

    label1.save("../dataset/ISBI/cut/{}_label_part1.png".format(index))
    label2.save("../dataset/ISBI/cut/{}_label_part2.png".format(index))
    label3.save("../dataset/ISBI/cut/{}_label_part3.png".format(index))
    label4.save("../dataset/ISBI/cut/{}_label_part4.png".format(index))
    index+=1