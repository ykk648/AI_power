from face_detect_and_align import FaceDetect
import cv2
import numpy as np
from ai_utils import img_show, img_save


if __name__ == '__main__':
    # === face detect and align ===
    fd = FaceDetect(mode='scrfd_500m')
    img_path = 'test_img/fake.jpeg'
    _, _ = fd.get_bboxes(img_path)
    # face_image, m_ = fd.get_single_face(crop_size=512, mode='default')
    face_image, m_ = fd.get_single_face(crop_size=512, mode='mtcnn_512')
    # face_image, m_ = fd.get_single_face(crop_size=512, mode='arcface_512')
    # face_image, ms = fd.get_single_face(crop_size=112, mode='arcface')
    img_show(face_image)
