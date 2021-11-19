# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
from PIL import Image
import numpy as np
import cv2

from utils import load_img_rgb, img_show
from .scrfd_insightface import SCRFD
from .mtcnn_pytorch import MTCNN
from face_detect_and_align.face_align_utils import norm_crop

# https://github.com/deepinsight/insightface/tree/master/detection/scrfd
SCRFD_MODEL_PATH = 'pretrain_models/face_detect/scrfd_onnx/'
# https://github.com/taotaonice/FaceShifter/blob/master/face_modules/mtcnn.py
# & https://github.com/TropComplique/mtcnn-pytorch
MTCNN_MODEL_PATH = 'pretrain_models/face_detect/mtcnn_weights/'


class FaceDetect5Landmarks:
    def __init__(self, mode='scrfd_500m'):
        self.mode = mode
        assert self.mode in ['scrfd', 'scrfd_500m', 'mtcnn']
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

    def get_bboxes(self, image, nms_thresh=0.5, max_num=0, min_bbox_size=None):
        """
        Args:
            image: RGB image path or Numpy array load by cv2
            nms_thresh:
            max_num:
            min_bbox_size:
        Returns:
        """
        self.image = load_img_rgb(image)
        if 'scrfd' in self.mode:
            self.bboxes, self.kpss = self.det_model_scrfd.detect_faces(self.image, thresh=nms_thresh, max_num=max_num,
                                                                       metric='max', min_face_size=64.0)
            if min_bbox_size is not None:
                self.bboxes_filter(min_bbox_size)
        else:
            pil_image = Image.fromarray(self.image)
            min_bbox_size = 64 if min_bbox_size is None else min_bbox_size
            self.bboxes, self.kpss = self.det_model_mtcnn.detect_faces(pil_image, min_face_size=min_bbox_size,
                                                                       thresholds=[0.6, 0.7, 0.8],
                                                                       nms_thresholds=[0.7, 0.7, 0.7])
        return self.bboxes, self.kpss

    def bboxes_filter(self, min_bbox_size):
        min_area = np.power(min_bbox_size, 2)
        area_list = (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
        min_index = np.where(area_list < min_area)
        self.bboxes = np.delete(self.bboxes, min_index, axis=0)
        self.kpss = np.delete(self.kpss, min_index, axis=0)

    def get_single_face(self, crop_size, mode='mtcnn_512'):
        """
        Args:
            crop_size:
            mode: default mtcnn_512 arcface_512 arcface default_95
        Returns: cv2 image
        """
        assert mode in ['default', 'mtcnn_512', 'arcface_512', 'arcface', 'default_95']
        if self.bboxes.shape[0] == 0:
            return None, None
        det_score = self.bboxes[..., 4]
        # select the face with the highest detection score
        best_index = np.argmax(det_score)
        kpss = None
        if self.kpss is not None:
            kpss = self.kpss[best_index]
        align_img, M = norm_crop(self.image, kpss, crop_size, mode=mode)
        align_img = cv2.cvtColor(align_img, cv2.COLOR_RGB2BGR)
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
