# -- coding: utf-8 --
# @Time : 2021/11/19
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box.utils import flush_print
import torch
import cv2
import torch.nn.functional as F
import numpy as np

GPEN_MODEL_PATH = 'pretrain_models/face_restore/gpen/GPEN-512.pth'


class GPEN:
    def __init__(self, size=512, use_gpu=True):
        flush_print('Start init Gpen model !')
        # op init costs time
        from face_lib.face_restore.gpen.face_gan import FaceGAN
        self.gpen = FaceGAN(model_path=GPEN_MODEL_PATH, size=size,
                            model='GPEN-512', channel_multiplier=2, use_gpu=use_gpu)
        flush_print('Gpen model init done !')

    def forward(self, img_):
        # print(type(img_))
        if type(img_) == str:
            img_ = cv2.imread(img_)
            return self.gpen.process(img_)
        elif type(img_) == torch.tensor:
            # for tensor
            with torch.no_grad():
                # img origin size
                # input_shape = img_.shape[-2:]
                face_tensor = img_ * 2.0 - 1.0
                face_tensor = face_tensor.unsqueeze(0)
                face_tensor = F.interpolate(face_tensor, size=(512, 512))

                enhanced, _ = self.gpen.model(face_tensor)

                enhanced = (enhanced + 1.0) / 2.0
                enhanced = torch.clip(enhanced, min=0.0, max=1.0)
                # enhanced = F.interpolate(enhanced, size=(output_size, output_size))
                return enhanced[0]
        elif isinstance(img_, np.ndarray):
            # read from opencv by default
            return self.gpen.process(img_)
