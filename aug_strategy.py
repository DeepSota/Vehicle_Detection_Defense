# filter.py
import cv2
import numpy as np
import math
import random



def compute(delta):
    k = round(3 * delta) * 2 + 1
    print('模的大小为:', k)
    H = np.zeros((k, k))
    k1 = (k - 1) / 2
    for i in range(k):
        for j in range(k):
            H[i, j] = (1 / (2 * 3.14 * (delta ** 2))) * math.exp((-(i - k1) ** 2 - (j - k1) ** 2) / (2 * delta ** 2))
    k3 = [k, H]
    print(H)
    print(sum(sum(H)))
    return k3



def relate(a, b, k):
    n = 0
    sum1 = np.zeros((k, k))

    for m in range(k):
        for n in range(k):
            sum1[m, n] = a[m, n] * b[m, n]
    return sum(sum(sum1))



def fil(imag, delta=0.7):
    k3 = compute(delta)
    k = k3[0]
    H = k3[1]
    k1 = (k - 1) / 2
    [a, b] = imag.shape
    k1 = int(k1)
    new1 = np.zeros((k1, b))
    new2 = np.zeros(((a + (k - 1)), k1))
    imag1 = np.r_[new1, imag]
    imag1 = np.r_[imag1, new1]
    imag1 = np.c_[new2, imag1]
    imag1 = np.c_[imag1, new2]
    y = np.zeros((a, b))
    sum2 = sum(sum(H))
    for i in range(k1, (k1 + a)):
        for j in range(k1, (k1 + b)):
            y[(i - k1), (j - k1)] = relate(imag1[(i - k1):(i + k1 + 1), (j - k1):(j + k1 + 1)], H, k) / sum2
    return y


def color_bgr2gray(img):
    row = img.shape[0]
    col = img.shape[1]
    for r in range(row):
        for l in range(col):
            img[r, l] = round(0.11 * img[r, l, 0] + 0.59 * img[r, l, 1] + 0.3 * img[r, l, 2])

    return img


