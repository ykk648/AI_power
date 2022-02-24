# -- coding: utf-8 --
# @Time : 2022/2/23
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import mediapipe as mp
import numpy as np
from cv2box import CVImage


class HandDetectorMediapipe:
    def __init__(self):
        # mp_drawing = mp.solutions.drawing_utils
        # mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_holistic = mp.solutions.holistic

    def forward(self, image):

        height, width = image.shape[0], image.shape[1]

        with self.mp_holistic.Holistic(
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            results = holistic.process(image)

            left_hand = results.left_hand_landmarks
            right_hand = results.right_hand_landmarks
            results = []
            for hand in [right_hand, left_hand]:
                if hand is not None:
                    hand_np = []
                    for i in range(21):
                        hand_np.append(
                            [hand.landmark[i].x * width, hand.landmark[i].y * height, hand.landmark[i].z * width])
                    box_left_top_x = np.min(hand_np, axis=0)[0]
                    box_left_top_y = np.min(hand_np, axis=0)[1]
                    box_right_bottle_x = np.max(hand_np, axis=0)[0]
                    box_right_bottle_y = np.max(hand_np, axis=0)[1]
                    results.append([box_left_top_x, box_left_top_y, box_right_bottle_x, box_right_bottle_y])
        return np.array(results)


if __name__ == '__main__':
    image_in = CVImage('').rgb
    # [[1113.7602996826172, 539.147379398346, 1374.1822814941406, 850.5021500587463]]
    # CVImage('').show()
    hdm = HandDetectorMediapipe()
    print(hdm.forward(image_in))
