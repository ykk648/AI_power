# -- coding: utf-8 --
# @Time : 2022/3/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch
"""
import cv2
import numpy as np
from cv2box import CVImage
import math
from model_convert.onnx_model import ONNXModel

model_path = 'pretrain_models/digital_human/body_detector_lightweight/body_detector_dynamic.onnx'


class BodyDetectorLightweight:
    def __init__(self, input_height_size=256, pad_value=(0, 0, 0), stride=8, upsample_ratio=4):
        self.input_height_size = input_height_size
        self.pad_value = pad_value
        self.stride = stride
        self.upsample_ratio = upsample_ratio
        self.input_std = 256
        self.input_mean = 128

        self.model = ONNXModel(model_path)

    @staticmethod
    def pad_width(img, stride, pad_value, min_dims):
        h, w, _ = img.shape
        h = min(min_dims[0], h)
        min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
        min_dims[1] = max(min_dims[1], w)
        min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
        pad = []
        pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
        pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
        pad.append(int(min_dims[0] - h - pad[0]))
        pad.append(int(min_dims[1] - w - pad[1]))
        padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3],
                                        cv2.BORDER_CONSTANT, value=pad_value)
        return padded_img, pad

    def forward(self, img):
        height, width, _ = img.shape
        scale = self.input_height_size / height
        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        min_dims = [self.input_height_size, max(scaled_img.shape[1], self.input_height_size)]
        padded_img, pad = self.pad_width(scaled_img, self.stride, self.pad_value, min_dims)

        stages_output = self.model.forward(
            CVImage(padded_img).set_blob(self.input_std, self.input_mean, input_size=None).blob_rgb)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps[0], (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio,
                              interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs

if __name__ == '__main__':
    img_p = 'test_img/t_pose.jpeg'
    bdl = BodyDetectorLightweight()
    results = bdl.forward(CVImage(img_p).bgr)
    print(results)
