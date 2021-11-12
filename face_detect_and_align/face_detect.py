# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from PIL import Image
import numpy as np
import cv2

from ai_utils import MyTimer, load_img, img_show
from scrfd_insightface import SCRFD
from mtcnn_pytorch import MTCNN
from face_detect_and_align.face_align_func import norm_crop

# https://github.com/deepinsight/insightface/tree/master/detection/scrfd
SCRFD_MODEL_PATH = 'pretrain_models/face_detect/scrfd_onnx/'
# https://github.com/taotaonice/FaceShifter/blob/master/face_modules/mtcnn.py
# & https://github.com/TropComplique/mtcnn-pytorch
MTCNN_MODEL_PATH = 'pretrain_models/face_detect/mtcnn_weights/'


class FaceDetect:
    def __init__(self, mode='scrfd_500m'):
        self.mode = mode
        assert self.mode in ['scrfd', 'scrf_500m', 'mtcnn']
        self.bboxes = self.kpss = self.image = None
        if 'scrfd' in self.mode:
            if self.mode == 'scrfd_500m':
                scrfd_model_path = SCRFD_MODEL_PATH + 'scrfd_500m_bnkps_shape640x640.onnx'
            else:
                scrfd_model_path = SCRFD_MODEL_PATH + 'scrfd_10g_bnkps.onnx'
            self.det_model_scrfd = SCRFD(scrfd_model_path)
            self.det_model_scrfd.prepare(ctx_id=0, input_size=(640, 640))
        elif self.mode == 'mtcnn':
            self.det_model_mtcnn = MTCNN(model_dir=MTCNN_MODEL_PATH)

    def get_bboxes(self, image, nms_thresh=0.5, max_num=0):
        """
        Args:
            image: image path or Numpy array load by cv2
            nms_thresh:
            max_num:
        Returns:
        """
        self.image = load_img(image)
        if 'scrfd' in self.mode:
            self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(self.image, thresh=nms_thresh, max_num=max_num,
                                                                       metric='max')
        else:
            pil_image = Image.fromarray(self.image)
            self.bboxes, self.kpss = self.det_model_mtcnn.detect_faces(pil_image, min_face_size=64.0,
                                                                       thresholds=[0.6, 0.7, 0.8],
                                                                       nms_thresholds=[0.7, 0.7, 0.7])

        return self.bboxes, self.kpss

    def get_single_face(self, crop_size, mode='mtcnn_512'):
        """
        Args:
            crop_size:
            mode: default mtcnn_512 arcface_512 arcface
        Returns:
        """
        assert mode in ['default', 'mtcnn_512', 'arcface_512', 'arcface']
        if self.bboxes.shape[0] == 0:
            return None
        det_score = self.bboxes[..., 4]
        # select the face with the highest detection score
        best_index = np.argmax(det_score)
        kpss = None
        if self.kpss is not None:
            kpss = self.kpss[best_index]
        align_img, M = norm_crop(self.image, kpss, crop_size, mode=mode)
        return align_img, M

    def get_multi_face(self, crop_size, mode='mtcnn_512'):
        """
        Args:
            crop_size:
            mode: default mtcnn_512 arcface_512 arcface
        Returns:
        """
        if self.bboxes.shape[0] == 0:
            return None
        align_img_list = []
        M_list = []
        for i in range(self.bboxes.shape[0]):
            kps = None
            if self.kpss is not None:
                kps = self.kpss[i]
            align_img, M = norm_crop(self.image, kps, crop_size, mode=mode)
            align_img_list.append(align_img)
            M_list.append(M)
        return align_img_list, M_list

    def draw_face(self):
        for i_ in range(self.bboxes.shape[0]):
            bbox = self.bboxes[i_]
            x1, y1, x2, y2, score = bbox.astype(int)
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if self.kpss is not None:
                kps = self.kpss[i_]
                for kp in kps:
                    kp = kp.astype(int)
                    cv2.circle(self.image, tuple(kp), 1, (0, 0, 255), 2)
        img_show(self.image)


if __name__ == '__main__':
    # # === face detect speed test and result show ===
    # fd = FaceDetect(mode='mtcnn')
    # img_path = 'test_img/fake.jpeg'
    # with MyTimer() as mt:
    #     # 3.47s
    #     for i in range(100):
    #         bboxes_mtcnn, kpss_mtcnn = fd.get_bboxes(img_path)
    # # print(bboxes, kpss)
    #
    # fd = FaceDetect(mode='scrfd_500m')
    # img_path = 'test_img/fake.jpeg'
    # with MyTimer() as mt:
    #     # 1.5s
    #     for i in range(100):
    #         bboxes_scrfd, kpss_scrfd = fd.get_bboxes(img_path)
    # # print(bboxes_scrfd, kpss_scrfd)
    # # fd.draw_face()

    # === face detect and align ===
    fd = FaceDetect(mode='scrfd_500m')
    img_path = 'test_img/fake.jpeg'
    _, _ = fd.get_bboxes(img_path)
    # face_image, m_ = fd.get_single_face(crop_size=512, mode='default')
    face_image, m_ = fd.get_single_face(crop_size=512, mode='mtcnn_512')
    # face_image, m_ = fd.get_single_face(crop_size=512, mode='arcface_512')
    # face_image, ms = fd.get_single_face(crop_size=112, mode='arcface')
    img_show(face_image)
