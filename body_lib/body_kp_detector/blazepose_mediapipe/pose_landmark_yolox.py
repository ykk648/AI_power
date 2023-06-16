# -- coding: utf-8 --
# @Time : 2022/7/21
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
from cv2box import CVImage, MyFpsCounter, CVBbox

from apstone import ONNXModel
from body_lib.body_kp_detector.blazepose_mediapipe.blaze_utils import postprocess, denormalize_landmarks, detection2roi, \
    extract_roi
from body_lib.body_bbox_detector import BodyBboxDetector

# input 1*256*256*3 output , 1*1 , , ,
LITE_MODEL = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/pose_landmark_lite.onnx'
FULL_MODEL = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/pose_landmark_full.onnx'
HEAVY_MODEL = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/pose_landmark_heavy.onnx'


class LandmarkDetectorYolox:
    def __init__(self, model_complexity=0, provider='gpu'):
        self.bbd = BodyBboxDetector(model='yolox_tiny_trt16', threshold=0.5)

        model_path_list = [LITE_MODEL, FULL_MODEL, HEAVY_MODEL]
        self.model = ONNXModel(model_path_list[model_complexity], provider=provider)

        self.need_bbox_flag = True
        self.history = []

    def forward(self, image_in_, show=False):
        """

        Args:
            image_in_:
            show:
        Returns:
            landmarks: 33*4

        """
        bbox_result = self.bbd.forward(image_in_, show=False, max_bbox_num=1)[0]
        img, ratio, left, top = CVImage(image_in_).crop_keep_ratio(bbox_result, (256, 256), padding_ratio=1.)

        if show:
            CVImage(img).show(0, 'img_crop')

        blob = (img / 256).astype(np.float32)[np.newaxis, :]
        normalized_landmarks, f, ee, rr, tt = self.model.forward(blob)
        normalized_landmarks = postprocess(normalized_landmarks)[0]
        landmarks_ = CVImage(None).recover_from_crop(normalized_landmarks, ratio, left, top, (256, 256))

        if show:
            show_img = CVImage(image_in_).draw_landmarks(landmarks_)
            CVImage(show_img).show(0, 'results')
        return landmarks_

    def forward_w_tracking(self, image_in_, show=False):
        if self.need_bbox_flag:
            bbox_result = self.bbd.forward(image_in_, show=False, max_bbox_num=1)[0]
        else:
            reserve_points = [0, 7, 8, 11, 12, 23, 24, 25, 26, 27, 28]
            bbox_result = CVBbox(None).get_bbox_from_points(self.history[-1][reserve_points], image_in_.shape,
                                                            margin_ratio=0.2)

        img, ratio, left, top = CVImage(image_in_).crop_keep_ratio(bbox_result, (256, 256), padding_ratio=1.)

        if show:
            CVImage(img).show(0, 'img_crop')

        blob = (img / 256).astype(np.float32)[np.newaxis, :]
        normalized_landmarks, f, _, _, _ = self.model.forward(blob)
        normalized_landmarks = postprocess(normalized_landmarks)[0]
        landmarks_ = CVImage(None).recover_from_crop(normalized_landmarks, ratio, left, top, (256, 256))

        self.need_bbox_flag = False
        self.history.append(landmarks_)
        self.history = self.history[-2:]

        if show:
            show_img = CVImage(image_in_).draw_landmarks(landmarks_)
            CVImage(show_img).show(0, 'results')
        return landmarks_


if __name__ == '__main__':
    # image_path = 'resource/for_pose/t_pose_1080p.jpeg'
    # image_in = CVImage(image_path).bgr

    """
    model 1 82fps trt16  trt 109fps
    model 2 67fps trt16 output Nan trt 97fps
    """
    ld = LandmarkDetectorYolox(model_complexity=2, provider='trt')

    # landmarks = ld.forward(image_in, show=True)
    # print(landmarks)
    #
    # with MyFpsCounter('model forward 10 times fps: ') as mfc:
    #     for i in range(10):
    #         filtered_detections = ld.forward(image_in)

    # video tracking test
    from cv2box import CVVideoLoader
    from tqdm import tqdm

    with CVVideoLoader('') as cvvl:
        for _ in tqdm(range(len(cvvl))):
            _, frame = cvvl.get()
            landmarks = ld.forward_w_tracking(frame, show=False)
