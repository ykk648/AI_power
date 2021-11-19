# -- coding: utf-8 --
# @Time : 2021/11/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import cv2
from utils.image_io import img_save, img_show
from .gpen import GPEN
from .dfdnet import DFDNet


class FaceRestore:
    def __init__(self, use_gpu=True, mode='gpen', verbose=True):
        self.use_gpu = use_gpu
        self.mode = mode
        self.face_result = None
        self.verbose = verbose
        if self.mode == 'gpen':
            self.fr = GPEN(size=512, use_gpu=self.use_gpu)
        elif self.mode == 'dfdnet':
            self.fr = DFDNet(use_gpu=self.use_gpu)

    def forward(self, img_, output_size=256):
        """
        Args:
            img_: cv2 BGR image or image path
            output_size: output image size
        Returns: cv2 BGR image
        """
        self.face_result = self.fr.forward(img_)
        return cv2.resize(self.face_result, (output_size, output_size))

    def save(self, img_save_p):
        img_save(self.face_result, img_save_p, self.verbose)
