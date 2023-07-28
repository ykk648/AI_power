# -- coding: utf-8 --
# @Time : 2022/7/21
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
"""
based on
https://github.com/PINTO0309/PINTO_model_zoo/tree/main/053_BlazePose
https://github.com/positive666/mediapipe_PoseEstimation_pytorch/blob/main/blazebase.py
https://github.com/Azzallon/teste/tree/DPR/pose_estimation_3d/blazepose-fullbody
"""

import numpy as np
from apstone import ONNXModel

from cv2box import CVImage, MyFpsCounter
from body_lib.body_kp_detector.blazepose_mediapipe.blaze_utils import denormalize_detections, \
    resize_pad, raw_output_to_detections, weighted_non_max_suppression

# from body_lib.body_kp_detector.blazepose_mediapipe.utils.blazepose_utils_numpy import raw_output_to_detections, \
#     weighted_non_max_suppression

# ANCHORS_128 = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/anchors/anchors_896_128.npy'
ANCHORS_224 = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/anchors/anchors_2254_224.npy'

# input: 1*3*224*224  output: score 1*2254*1 box 1*2254*12
LITE_BLAZEPOSE_MODEL = 'pretrain_models/body_lib/body_kp_detector/blazepose_mediapipe/densify_full_body_detector.onnx'


class BodyDetector:
    def __init__(self, provider='gpu'):
        super().__init__()
        self.anchors = np.load(ANCHORS_224)
        self.model = ONNXModel(LITE_BLAZEPOSE_MODEL, provider=provider)

        # self.input_std = 127.5
        # self.input_mean = 127.5
        # self.input_size = (224, 224)
        #
        # self.x_scale = self.y_scale = 224
        # self.w_scale = self.h_scale = 224
        # self.num_keypoints = 4
        # self.score_clipping_thresh = 100.0
        self.min_score_thresh = 0.5
        # self.min_suppression_threshold = 0.3
        # self.num_coords = 12

        # # These settings are for converting detections to ROIs which can then
        # # be extracted and feed into the landmark network
        # # use mediapipe/calculators/util/alignment_points_to_rects_calculator.cc
        # self.detection2roi_method = 'alignment'
        # self.kp1 = 2
        # self.kp2 = 3
        # self.theta0 = 90 * np.pi / 180
        # self.dscale = 1.5
        # self.dy = 0.

    def forward(self, img_in_, show=False):
        img_crop, scale, pad = resize_pad(CVImage(img_in_).bgr)
        image_blob = img_crop.astype(np.float32) / 255

        out = self.model.forward(image_blob.transpose((2, 1, 0))[np.newaxis, :])

        detections = raw_output_to_detections(out[1], out[0], self.anchors, self.min_score_thresh)

        filtered_detections = []
        for i in range(len(detections)):
            # faces = self._weighted_non_max_suppression(detections[i])
            faces = weighted_non_max_suppression(detections[i])
            faces = np.stack(faces) if len(faces) > 0 else np.zeros((0, 13))
            filtered_detections.append(faces)

        filtered_detections = denormalize_detections(filtered_detections[0], scale, pad)

        if show and len(filtered_detections) > 0:
            print(filtered_detections)
            # kps
            image_show = CVImage(img_in_).draw_landmarks(filtered_detections[0, 4:12].reshape((4, 2))[:, ::-1])
            # box
            image_show = CVImage(image_show).draw_landmarks(filtered_detections[0, 0:4].reshape((2, 2)),
                                                            color=(0, 255, 255))
            CVImage(image_show).show(0)
        return filtered_detections


if __name__ == '__main__':
    image_path = 'resources/yoga2.webp'
    image_in = CVImage(image_path).bgr
    pd = BodyDetector(provider='gpu')
    filtered_detections = pd.forward(image_in, show=True)

    # with MyFpsCounter('model forward 10 times fps: ') as mfc:
    #     for i in range(10):
    #         filtered_detections = pd.forward(image_in)

    # img_in, ratio, pad_w, pad_h = CVImage('resources/t_pose.jpeg').resize_keep_ratio((128, 128))
    # CVImage(img_in).show(0)
