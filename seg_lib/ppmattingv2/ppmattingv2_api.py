# -- coding: utf-8 --
# @Time : 2023/2/3
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from apstone import ModelBase
import cv2
import numpy as np
import copy
from cv2box import CVImage

MODEL_ZOO = {
    # https://github.com/jiachen0212/pp_mattingv2
    '384x480': {
        'model_path': 'pretrain_models/seg_lib/ppmattingv2/ppmattingv2_stdc1_human_384x480.onnx'
    },
}


class PPMattingV2(ModelBase):
    def __init__(self, model_name='384x480', provider='gpu'):
        super(PPMattingV2, self).__init__(MODEL_ZOO[model_name], provider)
        self.conf_threshold = 0.65

    def prepare_input(self, image):
        input_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=(self.input_width, self.input_height))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def forward(self, image):
        input_image = self.prepare_input(image)

        # Perform inference on the image
        result = self.forward(input_image)

        # Post process:squeeze
        segmentation_map = result[0]
        segmentation_map = np.squeeze(segmentation_map)

        image_width, image_height = image.shape[1], image.shape[0]
        dst_image = copy.deepcopy(image)
        segmentation_map = cv2.resize(
            segmentation_map,
            dsize=(image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # color list
        color_image_list = []
        # ID 0:BackGround
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 0)
        color_image_list.append(bg_image)
        # ID 1:Human
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 255, 0)
        color_image_list.append(bg_image)

        mask = np.where(segmentation_map > self.conf_threshold, 0, 1)
        mask = np.stack((mask,) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, dst_image, color_image_list[1])
        dst_image = cv2.addWeighted(dst_image, 0.5, mask_image, 0.5, 1.0)
        return dst_image


if __name__ == '__main__':
    ppm = PPMattingV2()
    img_p = ''
    ppm.forward(CVImage(img_p).bgr)
