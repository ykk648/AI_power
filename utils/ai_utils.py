import cv2
import PIL
import torch.nn.functional as F
import time
import numpy as np
import uuid
from pathlib import Path


def make_random_name(f_name):
    return uuid.uuid4().hex + '.' + f_name.split('.')[-1]


def down_sample(target_, size):
    return F.interpolate(target_, size=size, mode='bilinear', align_corners=True)


class MyTimer(object):
    """
    timer
    """

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print('[finished, spent time: {time:.2f}s]'.format(time=time.time() - self.t0))


def get_path_by_ext(this_dir, ext_list=None):
    if ext_list is None:
        print('Use image ext as default !')
        ext_list = [".jpg", ".png", ".JPG", ".webp", ".jpeg"]
    return [p for p in Path(this_dir).rglob('*') if p.suffix in ext_list]
