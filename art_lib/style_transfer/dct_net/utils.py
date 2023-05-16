# -- coding: utf-8 --
# @Time : 2023/1/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
import cv2


def resize_size(image, size=720):
    h, w, c = np.shape(image)
    if min(h, w) > size:
        if h > w:
            h, w = int(size * h / w), size
        else:
            h, w = size, int(size * w / h)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return image


def padTo16x(image):
    h, w, c = np.shape(image)
    if h % 16 == 0 and w % 16 == 0:
        return image, h, w
    nh, nw = (h // 16 + 1) * 16, (w // 16 + 1) * 16
    img_new = np.ones((nh, nw, 3), np.uint8) * 255
    img_new[:h, :w, :] = image

    return img_new, h, w
