# -- coding: utf-8 --
# @Time : 2021/11/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import cv2
import numpy as np
from PIL import Image

"""
skimage and pillow read image based uint8 and RGB mode
opencv read image based uint8 and BGR mode
using opencv as the default image read method
"""


def load_img_rgb(image):
    if type(image) == str:
        try:
            # image = Image.open(image).convert('RGB')
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        except cv2.error:
            raise Exception('Image path is empty or Image format do not support!')
    elif type(image) == np.ndarray:
        print('Got np array, assert its cv2 output.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_img_bgr(image):
    if type(image) == str:
        try:
            image = cv2.imread(image)
        except cv2.error:
            raise Exception('Image path is empty or Image format do not support!')
    elif type(image) == np.ndarray:
        print('Got np array, assert its cv2 output.')
    return image


def img_show(img, verbose=True):
    if verbose:
        print('img_format: {}'.format(type(img)))
    if type(img) is Image.Image:
        img.show()
    elif type(img) is str:
        cv2.imshow('test', cv2.imread(img))
        cv2.waitKey(0)
    else:
        cv2.imshow('test', img)
        cv2.waitKey(0)


def img_save(img, img_save_p, verbose=True):
    if verbose:
        print('img_format: {}'.format(type(img)))
    if type(img) is Image.Image:
        img.save(img_save_p, quality=95)
    else:
        cv2.imwrite(img_save_p, img)
