# -- coding: utf-8 --
# @Time : 2021/11/23
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from .utils import GFPGANer
from utils.image_io import load_img_bgr


class GFPGAN:
    def __init__(self, use_gpu=True):
        self.gpu = use_gpu

        self.model = GFPGANer(
            model_path='pretrain_models/face_restore/gfpgan/GFPGANCleanv1-NoCE-C2.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None)

    def forward(self, img_):
        """
        Args:
            img_: BGR image or path
        Returns: BGR image
        """
        if type(img_) == str:
            image_bgr = load_img_bgr(img_)
        else:
            image_bgr = img_
        try:
            cropped_face = self.model.enhance_single_aligned_image(image_bgr)
            return cropped_face
        except Exception as e:
            print('\t################ Error in enhancing this image: {}'.format(str(e)))
            raise e
