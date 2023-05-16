import os
# import numba
import numpy as np
from apstone import ModelBase
from cv2box import CVImage
from scipy.spatial import ConvexHull
import cv2

"""
ref https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model
    https://github.com/TalkUHulk/Image-Animation-Turbo-Boost
"""

MODEL_ZOO = {
    # input_name:['source'], shape:[[1, 3, 256, 256]]
    # output_name:['kp_source'], shape:[[1, 50, 2]]
    'kp_detector': {
        'model_path': 'pretrain_models/art_lib/talking_head/tpsmm/kp_detector.onnx',
    },
    # input_name: ['kp_source', 'source', 'kp_driving'], shape: [[1, 50, 2], [1, 3, 256, 256], [1, 50, 2]]
    # output_name: ['output'], shape: [[1, 3, 256, 256]]
    'tpsmm': {
        'model_path': 'pretrain_models/art_lib/talking_head/tpsmm/tpsmm.onnx',
    },
}


# def compute_hull_volume(hull):
#     """
#     计算凸包的体积
#     """
#     # 计算凸包缺陷
#     defects = cv2.convexityDefects(hull, np.arange(hull.shape[0]))
#
#     # 计算三角形面积并累加
#     volume = 0
#     for i in range(defects.shape[0]):
#         s, e, f, d = defects[i, 0]
#         start = tuple(hull[s][0])
#         end = tuple(hull[e][0])
#         far = tuple(hull[f][0])
#         a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
#         b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
#         c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
#         s = (a + b + c) / 2
#         area = np.sqrt(s * (s - a) * (s - b) * (s - c))
#         volume += area
#
#     print('凸包体积为：', volume)
#     # 请注意，此方法假定点集是二维的。对于三维点集，需要使用三角形面积的三维版本来计算体积。
#     return volume


def relative_kp(kp_source, kp_driving, kp_driving_initial):
    source_area = ConvexHull(kp_source[0]).volume
    # source_area = compute_hull_volume(cv2.convexHull(kp_source[0],returnPoints=False))
    driving_area = ConvexHull(kp_driving_initial[0]).volume
    # driving_area = compute_hull_volume(cv2.convexHull(kp_driving_initial[0], returnPoints=False))
    adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)

    kp_value_diff = (kp_driving - kp_driving_initial)
    kp_value_diff *= adapt_movement_scale
    kp_new = kp_value_diff + kp_source
    return kp_new


class KPDetector(ModelBase):
    def __init__(self, model_type='kp_detector', provider='cpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type

    def forward(self, img_source):
        """
        Args:
            img_source:
        Returns: 1, 50, 2
        """
        img_in = CVImage(img_source).blob((256, 256), input_mean=0, input_std=255, rgb=True)
        kp_results = self.model.forward(img_in)[0]
        return kp_results


class TPSMM(ModelBase):
    def __init__(self, model_type='tpsmm', provider='cpu'):
        super().__init__(MODEL_ZOO[model_type], provider)
        self.model_type = model_type
        self.kpd = KPDetector(provider='gpu')
        self.kp_driving_initial = None
        self.kp_source = None

    def get_kp_source(self, img_source):
        self.kp_source = self.kpd.forward(img_source)

    def forward(self, img_source, img_driving, pass_drive_kp=False):
        """
        Args:
            img_source:
            img_driving:
            pass_drive_kp: pass drive_kp instead of drive_img to speedup
        Returns:
        """
        if self.kp_source is None:
            raise 'Use get_kp_source func first to init kp_source !'

        img_source_in = CVImage(img_source).blob((256, 256), input_mean=0, input_std=255, rgb=True)
        # img_driving_in = CVImage(img_driving).blob((256, 256), input_mean=0, input_std=255, rgb=True)

        if self.kp_driving_initial is None:
            if not pass_drive_kp:
                self.kp_driving_initial = self.kpd.forward(img_driving)
            else:
                self.kp_driving_initial = img_driving

        if not pass_drive_kp:
            driving_kp = self.kpd.forward(img_driving)
        else:
            driving_kp = img_driving
        # relative
        kp_driving = relative_kp(self.kp_source, driving_kp, self.kp_driving_initial)
        # standard
        # kp_driving = self.kpd.forward(img_driving)

        drive_results = self.model.forward([self.kp_source, img_source_in, kp_driving])[0]
        output_img = drive_results[0].transpose(1, 2, 0)
        output_img = CVImage(output_img).rgb()  # reverse to bgr
        return output_img


if __name__ == '__main__':
    img_source_ = 'resource/cropped_face/112.png'
    img_driving_ = 'resource/cropped_face/512.jpg'

    tpsmm = TPSMM(provider='gpu')
    tpsmm.get_kp_source(img_source_)
    out_img = tpsmm.forward(img_source_, img_driving_)
    CVImage(out_img).show()
