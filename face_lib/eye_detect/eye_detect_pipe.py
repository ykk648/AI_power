# -- coding: utf-8 --
# @Time : 2023/5/15
# @Author : ykk648
# @Project : https://github.com/ykk648/eye_detect
"""
wrote by ChatGPT
"""
import cv2
import mediapipe as mp
import numpy as np
from face_lib.eye_detect.eye_open import EyeOpen
from cv2box import CVImage

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture('./test.mp4')
eo = EyeOpen()
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        # 读取每一帧图像
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # 转换颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h2, w2, c2 = image.shape
        # 处理当前帧图像
        results = face_mesh.process(image)
        left_eyes = []
        right_eyes = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    if idx in [465, 446, 348, 334, 446, 265]:  # left
                        x1 = np.int(landmark.x * w2)
                        y1 = np.int(landmark.y * h2)
                        left_eyes.append((x1, y1))
                    if idx in [52, 35, 230, 245, 66, 119]:  # left
                        x1 = np.int(landmark.x * w2)
                        y1 = np.int(landmark.y * h2)
                        right_eyes.append((x1, y1))

                right_box = cv2.boundingRect(np.asarray(right_eyes))
                left_box = cv2.boundingRect(np.asarray(left_eyes))
                # print(right_box, left_box)
                left_x1, left_y1, left_x2, left_y2 = left_box[0], right_box[1], left_box[0] + left_box[2], left_box[1] + \
                                                     left_box[3]
                # left_x1,left_y1,left_x2,left_y2 = left_box[0], right_box[1],left_box[0] + left_box[2], right_box[1] + right_box[3]

                # CVImage(image[left_y1:left_y2, left_x1:left_x2]).show()

                results = eo.forward(image[left_y1:left_y2, left_x1:left_x2])[0]
                print(results)
                if results[1] > 0.9:
                    info = 'open'
                elif results[0] > 0.9:
                    info = 'close'
                else:
                    info = 'open'

                # 绘制眼睛框
                cv2.rectangle(image, (right_box[0], right_box[1]),
                              (right_box[0] + right_box[2], right_box[1] + right_box[3]), (0, 255, 0), 2)
                cv2.rectangle(image, (left_box[0], left_box[1]), (left_box[0] + left_box[2], left_box[1] + left_box[3]),
                              (0, 255, 0), 2)

                # 绘制标签
                font = cv2.FONT_HERSHEY_SIMPLEX
                label = info
                font_size = 0.5
                thickness = 1
                color = (255, 255, 255)
                (label_width, label_height), _ = cv2.getTextSize(label, font, font_size, thickness)
                cv2.putText(image, label, (left_box[0], left_box[0] + left_box[2] + label_height),
                            font, font_size, color, thickness)

        # 转换颜色空间
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 显示当前帧图像
        CVImage(image).show()

# 释放摄像头
cap.release()
cv2.destroyAllWindows()
