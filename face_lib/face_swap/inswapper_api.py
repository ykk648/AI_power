# -- coding: utf-8 --
# @Time : 2023/7/26
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
from apstone import ModelBase
from cv2box import CVImage, CVFile

# Ref https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/inswapper.py

MODEL_ZOO = {
    # input_name: ['target', 'source'], shape: [[1, 3, 128, 128], [1, 512]]
    # output_name: ['output'], shape: [[1, 3, 128, 128]]
    'inswapper_128': {
        'model_path': 'pretrain_models/face_lib/face_swap/inswapper_128.onnx',
    },
}


class InSwapper(ModelBase):
    def __init__(self, model_name='inswapper_128', provider='gpu'):
        super().__init__(MODEL_ZOO[model_name], provider)
        self.emap = CVFile('pretrain_models/face_lib/face_swap/inswapper_emap.pkl').data

    def forward(self, src_face_image, dst_face_latent):
        """
        Args:
            src_face_image: RGB 0-255 128*128
            dst_face_latent: [1, 512]
        Returns: (1,3,128,128)
        """
        img_tensor = (src_face_image.transpose(2, 0, 1) / 255.0)[np.newaxis, ...]

        # dont know why, ref https://github.com/deepinsight/insightface/issues/2384
        dst_face_latent = np.dot(dst_face_latent, self.emap)
        dst_face_latent /= np.linalg.norm(dst_face_latent)

        blob = [img_tensor.astype(np.float32), dst_face_latent.astype(np.float32)]
        swap_face = self.model.forward(blob)
        return swap_face
