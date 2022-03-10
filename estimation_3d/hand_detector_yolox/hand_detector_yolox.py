# -- coding: utf-8 --
# @Time : 2022/1/7
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np
from cv2box import CVImage, MyFpsCounter
from model_convert.onnx_model import ONNXModel
import cv2

MODEL_PATH = 'pretrain_models/digital_human/hand_detecotr_yolox/yolox_100DOH_epoch90.onnx'


class HandDetectorYolox:
    def __init__(self, thres=0.5):
        self.model = ONNXModel(MODEL_PATH, debug=False)
        self.thres = thres

    def forward(self, image_p_, show=False):
        blob, ratio, pad_w, pad_h = CVImage(image_p_).resize_keep_ratio((640, 640), pad_value=(114, 114, 114))
        results, label = self.model.forward(np.float32(blob), trans=True)
        results_after = []
        img_origin = None
        for index, bbox in enumerate(results[0]):
            if bbox[4] > self.thres and label[0][index] == 1:
                bbox[0] = int((bbox[0] - pad_w // 2) / ratio)
                bbox[2] = int((bbox[2] - pad_w // 2) / ratio)
                bbox[1] = int((bbox[1] - pad_h // 2) / ratio)
                bbox[3] = int((bbox[3] - pad_h // 2) / ratio)
                results_after.append(bbox[:-1])
        hdy_result = np.array(results_after).astype(np.int)
        if show:
            img_origin = CVImage(image_p_).bgr.copy()
            for bbox_ in hdy_result:
                cv2.rectangle(img_origin, (bbox_[0], bbox_[1]), (bbox_[2], bbox_[3]), (0, 255, 0), 4)
            CVImage(img_origin).show(wait_time=1)
        return np.array(results_after).astype(np.int), img_origin


if __name__ == '__main__':
    # image_p = 'test_img/test1.jpg'

    cap = cv2.VideoCapture(0)
    hd = HandDetectorYolox(0.5)
    while True:
        success, img = cap.read()
        # fps 130
        with MyFpsCounter() as mfc:
            hd_result = hd.forward(img, show=False)
