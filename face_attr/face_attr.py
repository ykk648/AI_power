# -- coding: utf-8 --
# @Time : 2022/1/7
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
from cv2box import CVImage
from model_convert.onnx_model import ONNXModel

MODEL_PATH = './pretrain_models/face_attr/face_attr_epoch_12_220318.onnx'


class FaceAttr:
    def __init__(self):
        self.model = ONNXModel(MODEL_PATH, debug=False)

    def forward(self, image_p_):
        blob = CVImage(image_p_).resize((512, 512)) \
            .innormal(mean=[132.38155592, 110.99284567, 102.62942472], std=[68.5106407, 61.65929394, 58.61700102])
        result = self.model.forward(blob, trans=True)[0]
        return np.around(result, 3)

    @staticmethod
    def show_label():
        print('female male front side clean occlusion super_hq hq blur nonhuman')


if __name__ == '__main__':
    image_p = 'test_img/cropped_face/512.jpg'
    fa = FaceAttr()

    print(fa.forward(image_p))
