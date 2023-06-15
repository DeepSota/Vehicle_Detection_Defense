# via name
import random

import torch

from imagecorruptions import corrupt                   # 注意image需为numpy array类型
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import imgaug.augmenters as iaa

aug = []
aug.append(iaa.Canny(alpha=(0.0, 0.5), colorizer=iaa.RandomColorsBinaryImageColorizer(color_true=255, color_false=0)))
aug.append(iaa.Dropout((0.1, 0.5), per_channel=False))
aug.append(iaa.CoarseDropout((0.02, 0.10), size_percent=(0.15, 0.55), per_channel=False))

def showimg(img):
    plt.imshow(img)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()
# 必须有这个，要不然无法显示

def corrupt_im(image, prob):
    select_prob = random.random()
    if select_prob  > (1-prob):
        corrupt_list = np.arange(0, 18).tolist()
        del corrupt_list[4:7]
        cor_numb = random.sample(corrupt_list, 1)[0]
        sever_id = random.randint(2, 5)
        corrupted = corrupt(image, corruption_number=cor_numb, severity=sever_id)  # 模糊可以暂时去掉
        image = corrupted
    else:
        image
    return  image

def iaa_im(image, prob):
    select_prob = random.random()
    if select_prob > (1-prob):
        iaa_id = random.randint(0, len(aug)-1)  # 这里 len 要减1 避免超过
        image = aug[iaa_id].augment_image(image)
    else:
        image
    return  image





