# via name
import random
from imagecorruptions import corrupt
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np

def showimg(img):
    plt.imshow(img)
    plt.axis('on')
    plt.title('image')
    plt.show()



image = imread('data./images/bus.jpg')
showimg(image)


corrupted_image = corrupt(image, corruption_name='gaussian_blur', severity=1)
showimg(corrupted_image)


from imagecorruptions import get_corruption_names
corrupt_list = np.arange(0, 18).tolist()

del corrupt_list[3:7]
for q in range(len(corrupt_list)):
    i = corrupt_list[q]
    for severity in range(3, 5):
        corrupted = corrupt(image, corruption_number=i, severity=severity+1)
        showimg(corrupted)

