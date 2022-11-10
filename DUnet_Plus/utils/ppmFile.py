# -*- coding:utf-8 -*-
# author:peng
# Date：2022/9/7 15:40
import gzip

from PIL import Image
import glob

# files = glob.glob('../dataset/stare/*')
# print(files)
# for file in files:
#     filename = file.split('\\')[1].split('.')[0]
#     print(filename)
#     img = Image.open(file)
#     img.save('../dataset/stareChange/'+filename+'.jpg')

# file = '../dataset/stare-labels-ah/im0001.ah.ppm.gz'
# file_name = file.replace('.gz','')
# g_file = gzip.GzipFile(file)
# img = Image.open(g_file)
# img.show()
files = glob.glob('../dataset/stare-labels-ah/*')
for file in files:
    # print(file)
    filename = file.split("\\")[1].split('.ah')[0]
    print(filename)
    g_file = gzip.GzipFile(file)
    img = Image.open(g_file)
    img.save('../dataset/stareLabelChange/'+filename+'.jpg')


