#
#
#
# import os
# import shutil
#
# path = "E:\Desktop\FF\\02_手写字符\数据\data"
#
# data_name = os.listdir(path)
#
# path = "E:\Desktop\FF\\02_手写字符\数据\data\\"
#
# path_train = "E:\Desktop\FF\\02_手写字符\数据\data_use\\train"
# path_test = "E:\Desktop\FF\\02_手写字符\数据\data_use\\test\\"
#
# for i in data_name:
#     if int(i[-6:-4]) > 15:
#         path_temp = path + i
#


# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 16:28:13 2021

@author: NN
"""

import os
import random
import shutil

# 原始数据集路径
# origion_path = r'D:\蓝藻门'
origion_path = r'E:\Desktop\FF\08_窗帘缺陷\code_resnet18\data'
names = os.listdir(origion_path)

# 保存路径
# save_train_dir = r'D:\藻类识别神经网络\分类网络\train'
# save_test_dir = r'D:\藻类识别神经网络\分类网络\test'

# 数据集类别及数量
for i in names:
    file_list = origion_path + '\\' + i
    image_list = os.listdir(file_list)  # 获取图片的原始路径
    image_number = len(image_list)

    train_number = int(image_number * 0.9)
    train_sample = random.sample(image_list, train_number)  # 从image_list中随机获取0.8比例的图像.
    test_sample = list(set(image_list) - set(train_sample))

    # 创建保存路径
    save_train_dir = r'E:\Desktop\FF\08_窗帘缺陷\code_resnet18\data\\train' + '\\' + i
    save_test_dir = r'E:\Desktop\FF\08_窗帘缺陷\code_resnet18\data\\test' + '\\' + i
    if not os.path.isdir(save_train_dir):
        os.makedirs(save_train_dir)

    if not os.path.isdir(save_test_dir):
        os.makedirs(save_test_dir)

    # 复制图像到目标文件夹
    for j in train_sample:
        shutil.copy(file_list + '\\' + j, save_train_dir)

    for k in test_sample:
        shutil.copy(file_list + '\\' + k, save_test_dir)

pass