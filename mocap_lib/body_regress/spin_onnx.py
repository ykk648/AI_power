# -- coding: utf-8 --
# @Time : 2022/4/14
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import torch
from apstone import ONNXModel

# https://github.com/nkolot/SPIN

onnx_model_p = 'pretrain_models/body_regressor_spin/body_regressor_spin-eft-agora.onnx'

spin = ONNXModel(onnx_model_p)
print(spin.forward(torch.randn(1, 3, 224, 224).numpy()))
