import cv2
import PIL
import torch.nn.functional as F
import time
import numpy as np


def down_sample(target_, size):
    return F.interpolate(target_, size=size, mode='bilinear', align_corners=True)


def img_show(img, verbose=True):
    if verbose:
        print('img_format: {}'.format(type(img)))
    if type(img) is PIL.Image.Image:
        img.show()
    else:
        cv2.imshow('', img)
        cv2.waitKey(0)


def img_save(img, img_save_p, verbose=True):
    if verbose:
        print('img_format: {}'.format(type(img)))
    if type(img) is PIL.Image.Image:
        img.save(img_save_p, quality=95)
    else:
        cv2.imwrite(img_save_p, img)


class MyTimer(object):
    """
    timer
    """

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[finished, spent time: {time:.2f}s]'.format(time=time.time() - self.t0))


def load_img(image):
    if type(image) == str:
        image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        # image = Image.open(image).convert('RGB')
    elif type(image) == np.ndarray:
        print('Got np array, assert its cv2 output.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
