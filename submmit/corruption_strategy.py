# via name
import random

from imagecorruptions import corrupt

import matplotlib.pyplot as plt
import numpy as np

def showimg(img):
    plt.imshow(img)
    plt.axis('on')
    plt.title('image')
    plt.show()


def corrupt_im(image, prob):
    select_prob = random.random()
    if select_prob  > (1-prob):
        corrupt_list = np.arange(0, 18).tolist()
        del corrupt_list[3:7]
        cor_numb = random.sample(corrupt_list, 1)[0]
        sever_id = random.randint(2, 5)
        corrupted = corrupt(image, corruption_number=cor_numb, severity=sever_id)
        image = corrupted
    else:
        image
    return  image

def corrupt_im_single(image,id):
    mag = 3
    corrupt_list = [5,7,8,9,10,11,12,13,14]
    cor_numb = corrupt_list[id]
    corrupted = corrupt(image, corruption_number=cor_numb, severity=mag)
    image = corrupted

    return  image

def corrupt_single(image,id):
    mag = 3
    cor_numb = id
    corrupted = corrupt(image, corruption_number=cor_numb, severity=mag)
    image = corrupted

    return  image