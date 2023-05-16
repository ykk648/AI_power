# -- coding: utf-8 --
# @Time : 2022/5/13
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import cv2
from cv2box import CVImage, CVVideoLoader, CVFile

cam_ids = [0]

for cam_id in cam_ids:
    with CVVideoLoader(cam_id) as cvvl:
        cvvl.cap.set(3, 640)
        cvvl.cap.set(4, 480)
        print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(cvvl.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(cvvl.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = 0
        while True:
            if frame_count > 300:
                break
            _, frame = cvvl.get()
            CVImage(frame).show(1)
            if frame_count % 15 == 0:
                CVImage(frame).save('./cam{}/image{:02d}.jpg'.format(cam_id, frame_count // 15), create_path=True)
            frame_count += 1
