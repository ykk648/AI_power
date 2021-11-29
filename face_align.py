from face_detect_and_align import FaceDetect5Landmarks
import cv2
import numpy as np
from utils import img_show, img_save, get_path_by_ext, MyTimer
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':

    # # === face detect speed test and result show ===
    # fd = FaceDetect(mode='mtcnn')
    # img_path = 'test_img/fake.jp4g'
    # with MyTimer() as mt:
    #     # 3.47s
    #     for i in range(100):
    #         bboxes_mtcnn, kpss_mtcnn = fd.get_bboxes(img_path)
    # # print(bboxes, kpss)

    fd = FaceDetect5Landmarks(mode='scrfd_500m')
    img_path = 'test_img/test2.jpg'
    with MyTimer() as mt:
        # 1.5s
        for i in range(100):
            bboxes_scrfd, kpss_scrfd = fd.get_bboxes(img_path)
    # print(bboxes_scrfd, kpss_scrfd)
    fd.draw_face()

    # # === face detect and align ===
    # fd = FaceDetect5Landmarks(mode='scrfd_500m')
    # img_path = 'test_img/test1.jpg'
    # _, _ = fd.get_bboxes(img_path, min_bbox_size=64)
    # face_image, m_ = fd.get_single_face(crop_size=512, mode='default_95')
    # # face_image, m_ = fd.get_single_face(crop_size=512, mode='mtcnn_512')
    # # face_image, m_ = fd.get_single_face(crop_size=512, mode='arcface_512')
    # # face_image, ms = fd.get_single_face(crop_size=112, mode='arcface')
    # if face_image is not None:
    #     img_show(face_image)
    #     img_save(face_image, 'test_img/croped_face/test1.jpg')
    #     # img_show(img_path)

    # # == batch images face detect and align ==
    # fd = FaceDetect5Landmarks(mode='scrfd_500m')
    # img_dir = ''
    # exts = [".jpg", ".png", ".JPG", ".webp", ".jpeg"]
    #
    # for img_path in tqdm(list(get_path_by_ext(img_dir, exts))):
    #
    #     img_save_path = Path(str(img_path.with_suffix('.jpg')).replace('something', 'something_new'))
    #     img_save_path.parent.mkdir(parents=True, exist_ok=True)
    #     if not img_save_path.exists():
    #         try:
    #             _, _ = fd.get_bboxes(str(img_path))
    #             face_image, m_ = fd.get_single_face(crop_size=512, mode='default_95')  # default_95
    #             face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    #             img_save(face_image, str(img_save_path), verbose=False)
    #         except (TypeError, cv2.error):
    #             pass
