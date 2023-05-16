# -- coding: utf-8 --
# @Time : 2023/5/16
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
ref https://github.com/abhilb/Open-eye-closed-eye-classification
"""
import numpy as np
from cv2box import CVImage
from apstone import ONNXModel

"""
input_name:['input_1'], shape:[[1, 1, 32, 32]]
output_name:['Identity'], shape:[[1, 2]]
"""

MODEL_PATH = './pretrain_models/face_lib/eye_open_detect/open_close_eye_model.onnx'


class EyeOpen:
    def __init__(self):
        self.model = ONNXModel(MODEL_PATH, debug=False)

    def forward(self, image_p_):
        blob = CVImage(CVImage(image_p_).gray()).resize((32, 32)).bgr.reshape(1, 1, 32, 32).astype(np.float32) / 255
        result = self.model.forward(blob, trans=False)[0]
        return np.around(result, 3)


if __name__ == '__main__':
    # close [0,1] open [1,0]
    image_p = './resource/open.JPG'
    eo = EyeOpen()

    print(eo.forward(image_p))
