import os
# import numba
import numpy as np
from apstone import ModelBase
from cv2box import CVImage
from scipy.spatial import ConvexHull
import cv2

"""
ref https://github.com/OpenTalker/SadTalker/blob/main/src/audio2pose_models/cvae.py
"""

MODEL_ZOO = {
    # input: 32frames z(random) class(style) ref(reference coeff) audio_emb(audio feature)
    # input_name: ['input_1', 'input_2', 'input_3', 'input_4'], shape: [[1, 6], [1], [1, 64], [1, 32, 512]]
    # output: pose_motion_pred
    # output_name: ['output_1'], shape: [[1, 32, 6]]
    'audio2pose_decoder': {
        'model_path': 'pretrain_models/art_lib/talking_head/sadtalker/audio_2_pose.onnx',
    },
}


class Audio2PoseDecoder(ModelBase):
    def __init__(self, model_type='audio2pose_decoder', provider='cpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, img_source, img_driving, pass_drive_kp=False):
        pass


if __name__ == '__main__':
    pass
