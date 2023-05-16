# -- coding: utf-8 --
# @Time : 2022/6/6
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import numpy as np
import mediapipe as mp
from cv2box import CVImage, CVVideoLoader
import cv2
from tqdm import tqdm


class MediapipeHolistic:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            # model_complexity=2,
            smooth_landmarks=True,
            # refine_face_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def draw_show(self, image_in_, results):
        self.mp_drawing.draw_landmarks(
            image_in_,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
        self.mp_drawing.draw_landmarks(
            image_in_,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        self.mp_drawing.draw_landmarks(
            image_in_,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())
        self.mp_drawing.draw_landmarks(
            image_in_,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style())
        CVImage(image_in_).show(wait_time=1)
        # CVImage(image).save(
        #     '/{}.jpg'.format(
        #         i), create_path=True)

    @staticmethod
    def result_convert(results_in, image_in_shape):
        results_out = []
        for i in range(len(results_in)):
            # print(results_in[i].x)
            results_out.append([results_in[i].x * image_in_shape[1], results_in[i].y * image_in_shape[0], 1.])
        return results_out

    def forward(self, image_in_, draw_show=False):
        image_in_ = CVImage(image_in_).rgb
        image_in_.flags.writeable = False
        results = self.holistic.process(image_in_)
        image_in_.flags.writeable = True
        image_in_ = cv2.cvtColor(image_in_, cv2.COLOR_RGB2BGR)

        if draw_show:
            self.draw_show(image_in_, results)

        body_kp = self.result_convert(results.pose_landmarks.landmark, image_in_.shape)
        left_hd_kp = self.result_convert(results.left_hand_landmarks.landmark, image_in_.shape)
        right_hd_kp = self.result_convert(results.right_hand_landmarks.landmark, image_in_.shape)

        return body_kp, left_hd_kp, right_hd_kp


if __name__ == '__main__':
    mh = MediapipeHolistic()

    with CVVideoLoader(
            '') as cvvl:
        for i in tqdm(range(len(cvvl))):
            _, image = cvvl.get()
            body_kp, left_h, right_h = mh.forward(image)
            print(len(left_h))
