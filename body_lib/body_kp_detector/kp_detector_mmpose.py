# -- coding: utf-8 --
# @Time : 2022/8/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
https://modelscope.cn/models/damo/cv_hrnetv2w32_body-2d-keypoints_image/summary
"""
from cv2box import CVImage, MyFpsCounter
from apstone import KpDetectorBase
import cv2
import numpy as np

MODEL_ZOO = {
    # input_name:['input_1'], shape:[[1, 3, 128, 128]]
    # output_name:['output1'], shape:[[1, 15, 32, 32]]
    'hrnetv2w32': {
        'model_path': 'pretrain_models/body_lib/body_kp_detector/modelscope_hrnetv2w32.onnx',
        'model_input_size': (128, 128)
    },  # w h
}

class BodyDetectorModelScope(KpDetectorBase):
    def __init__(self, model_type='r50', provider='gpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.dark_flag = model_type.find('dark') > 0

    def forward(self, image_in_, bbox_, show=False, mirror_test=False):
        if len(bbox_) == 0:
            return [[0, 0, 0]] * 133

        model_results = self.model.forward(self.preprocess(image_in_, bbox_))

        kp_results = self.post_process_default(model_results[0], self.ratio, self.left, self.top)

        if show:
            self.show(image_in_, kp_results)

        return kp_results


if __name__ == '__main__':
    image_path = 'resource/for_pose/t_pose_1080p.jpeg'
    image_in = CVImage(image_path).bgr
    bbox = [493, 75, 1427, 1044]

    bwd = BodyDetectorModelScope(model_type='hrnetv2w32', provider='gpu')
    kps = bwd.forward(image_in, bbox, show=True, mirror_test=False)
    # print(kps)

    # with MyFpsCounter('model forward 10 times fps: ') as mfc:
    #     for i in range(10):
    #         kps = bwd.forward(image_in, bbox)

    # # for video
    # from cv2box import CVVideoLoader
    # from tqdm import tqdm
    #
    # with CVVideoLoader('') as cvvl:
    #     for _ in tqdm(range(len(cvvl))):
    #         _, frame = cvvl.get()
    #         kps = bwd.forward(image_in, bbox, show=True, mirror_test=False)
