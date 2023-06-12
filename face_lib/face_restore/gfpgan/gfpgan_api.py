# -- coding: utf-8 --
# @Time : 2021/11/23
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
https://github.com/TencentARC/GFPGAN
https://github.com/sczhou/CodeFormer
https://github.com/wzhouxiff/RestoreFormer
"""

from .gfpgan_utils import GFPGANer
from cv2box import CVImage


class GFPGAN:
    def __init__(self, use_gpu=True, version=2):
        self.gpu = use_gpu

        if version == 2:
            pretrain_model_path = 'pretrain_models/face_lib/face_restore/gfpgan/GFPGANCleanv1-NoCE-C2.pth'
        elif version == 4:
            pretrain_model_path = 'pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.4.pth'
        elif version == 'RestoreFormer':
            pretrain_model_path = 'pretrain_models/face_lib/face_restore/gfpgan/RestoreFormer.pth'
        elif version == 'CodeFormer':
            pretrain_model_path = 'pretrain_models/face_lib/face_restore/gfpgan/CodeFormer.pth'
        else:
            pretrain_model_path = 'pretrain_models/face_lib/face_restore/gfpgan/GFPGANv1.3.pth'

        if version not in ['CodeFormer', 'RestoreFormer']:
            arch = 'clean'
        else:
            arch = version

        self.model = GFPGANer(
            model_path=pretrain_model_path,
            upscale=2,
            arch=arch,
            channel_multiplier=2,
            bg_upsampler=None)

    def forward(self, img_):
        """
        Args:
            img_: BGR image or path
        Returns: BGR image
        """
        image_bgr = CVImage(img_).bgr
        try:
            cropped_face = self.model.enhance_single_aligned_image(image_bgr)
            return cropped_face
        except Exception as e:
            print('\t################ Error in enhancing this image: {}'.format(str(e)))
            raise e
