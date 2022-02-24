# -- coding: utf-8 --
# @Time : 2022/2/14
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from model_convert.onnx_model import ONNXModel
import numpy as np


class IKModel:
    def __init__(self):
        self.ik_model = ONNXModel('pretrain_models/digital_human/minimal_hands/iknet/iknet.onnx')

    def forward(self, pack):
        pack = np.expand_dims(pack, 0)
        theta = self.ik_model.forward(pack.astype(np.float32))[0]
        # theta_mano = mpii_to_mano(theta)
        if len(theta.shape) == 3:
            theta = theta[0]
        return theta
