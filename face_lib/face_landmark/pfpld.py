# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
Ref https://github.com/hanson-young/nniefacelib/tree/master/PFPLD/models/onnx
"""
import cv2
import onnxruntime as ort
import numpy as np

from cv2box import CVImage
from apstone import ModelBase
from face_lib.face_landmark.utils import convert98to68

MODEL_ZOO = {
    'pfpld': {
        'model_path': 'pretrain_models/face_lib/face_landmark/pfpld.onnx',
        'model_input_size': (112, 112), },
}


class PFPLD(ModelBase):
    def __init__(self, model_name='pfpld', provider='gpu', cpu=False):
        super().__init__(MODEL_ZOO[model_name], provider)

    def forward(self, face_image):
        """
        Args:
            face_image: RGB
        Returns:
        """
        input_image_shape = face_image.shape
        face_image = CVImage(face_image).resize((112, 112))
        face_image = (face_image / 255).astype(np.float32)
        pred = self.model.forward(face_image, trans=True)
        pred = convert98to68(pred[1])
        return pred.reshape(-1, 68, 2) * input_image_shape[:2][::-1]
