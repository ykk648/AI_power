# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from PIL import Image
import numpy as np
import cv2

from scrfd_insightface import SCRFD

SCRFD_MODEL_PATH = 'pretrain_models/face_detect/scrfd_onnx/'


class FaceDetect:
    def __init__(self, mode='scrfd_500m'):
        if 'scrfd' in mode:
            if mode == 'scrfd_500m':
                scrfd_model_path = SCRFD_MODEL_PATH + 'scrfd_500m_bnkps_shape640x640.onnx'
            else:
                scrfd_model_path = SCRFD_MODEL_PATH + 'scrfd_10g_bnkps.onnx'
            self.det_model = SCRFD(scrfd_model_path)
            self.det_model.prepare(ctx_id=0, input_size=(640, 640))
        elif mode == 'mtcnn':
            pass

    def get_bboxes(self, image, nms_thresh=0.5, max_num=0):
        if type(image) == str:
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        elif type(image) == np.ndarray:
            print('Got np array, assert its cv2 output.')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes_, kpss_ = self.det_model.detect(image, thresh=nms_thresh, max_num=max_num, metric='max')
        return bboxes_, kpss_


if __name__ == '__main__':

    fd = FaceDetect()
    img_path = 'test_img/fake.jpeg'
    bboxes, kpss = fd.get_bboxes(img_path)

    img = cv2.imread(img_path)

    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2, score = bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if kpss is not None:
            kps = kpss[i]
            for kp in kps:
                kp = kp.astype(int)
                cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
    filename = img_path.split('/')[-1]
    print('output:', filename)
    cv2.imwrite('./%s' % filename, img)
