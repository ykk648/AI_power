import os
import cv2
import numpy as np
from apstone import ModelBase
from cv2box import CVImage

from art_lib.style_transfer.dct_net.utils import resize_size, padTo16x

MODEL_ZOO = {
    '3d': {
        'model_path': 'pretrain_models/art_lib/style_transfer/dctnet/3d_h.onnx',
        'input_dynamic_shape': (720, 720, 3),
    },
    'anime': {
        'model_path': 'pretrain_models/art_lib/style_transfer/dctnet/anime_h.onnx',
        'input_dynamic_shape': (720, 720, 3),
    },
    'artstyle': {
        'model_path': 'pretrain_models/art_lib/style_transfer/dctnet/artstyle_h.onnx',
        'input_dynamic_shape': (720, 720, 3),
    },
    'handdrawn': {
        'model_path': 'pretrain_models/art_lib/style_transfer/dctnet/handdrawn_h.onnx',
        'input_dynamic_shape': (720, 720, 3),
    },
    'sketch': {
        'model_path': 'pretrain_models/art_lib/style_transfer/dctnet/sketch_h.onnx',
        'input_dynamic_shape': (720, 720, 3),
    },
}


class DCTNet(ModelBase):
    def __init__(self, model_type='anime', provider='cpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, img_in):
        # img: BGR input
        img_bgr = CVImage(img_in).bgr
        ori_h, ori_w, _ = img_bgr.shape
        img_bgr = resize_size(img_bgr, size=720).astype(np.float32)
        pad_bg, pad_h, pad_w = padTo16x(img_bgr)
        pad_bg = pad_bg.astype(np.float32)
        bg_res = self.model.forward(pad_bg)[0]
        res = bg_res[:pad_h, :pad_w, :]

        res = cv2.resize(res, (ori_w, ori_h), interpolation=cv2.INTER_AREA)
        res = np.clip(res, 0, 255).astype(np.uint8)
        return res


if __name__ == '__main__':
    image_p = 'resource/test3.jpg'
    dct = DCTNet(model_type='3d')
    out_img = dct.forward(image_p)
    CVImage(out_img).show()
