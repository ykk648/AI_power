# -- coding: utf-8 --
# @Time : 2023/7/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import os
# import numba
import numpy as np
from apstone import ModelBase
from cv2box import CVImage
from scipy.spatial import ConvexHull
import cv2
"""
input_name:['input_1'], shape:[[1, 32, 1, 80, 16]]
output_name:['output_1'], shape:[[1, 32, 512]]
"""

MODEL_ZOO = {
    # 32 frame , mel spectrogram
    # input_name: ['input_1'], shape: [[1, 32, 1, 80, 16]]
    # output_name: ['output_1'], shape: [[1, 32, 512]]
    'audio_encoder': {
        'model_path': 'pretrain_models/talking_head/wav2lip/audio_encoder.onnx',
    },
}


class Audio2PoseDecoder(ModelBase):
    def __init__(self, model_type='audio_encoder', provider='cpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, img_source, img_driving, pass_drive_kp=False):
        pass