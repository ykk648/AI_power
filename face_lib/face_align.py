from face_lib.face_detect_and_align import FaceDetect5Landmarks
import cv2
from cv2box import CVImage
from cv2box.utils.util import get_path_by_ext
from pathlib import Path
from tqdm import tqdm

if __name__ == '__main__':

    # # === face detect speed test and result show ===
    # fd = FaceDetect5Landmarks(mode='mtcnn')
    # img_path = 'test_img/test2.jpg'
    # with MyTimer() as mt:
    #     # 3.47s
    #     for i in range(100):
    #         bboxes_mtcnn, kpss_mtcnn = fd.get_bboxes(img_path)
    # # print(bboxes, kpss)

    # fd = FaceDetect5Landmarks(mode='scrfd_500m')
    # img_path = 'test_img/test2.jpg'
    # with MyTimer() as mt:
    #     # 1.5s
    #     for i in range(100):
    #         bboxes_scrfd, kpss_scrfd = fd.get_bboxes(img_path)
    # # print(bboxes_scrfd, kpss_scrfd)
    # fd.draw_face()

    # === face detect and align ===
    fd = FaceDetect5Landmarks(mode='scrfd_500m')
    img_path = ''
    bboxes_scrfd, kpss_scrfd = fd.get_bboxes(img_path, min_bbox_size=64)
    # fd.draw_face()
    # print(bboxes_scrfd)
    face_image, m_ = fd.get_single_face(crop_size=512, mode='mtcnn_512') # mtcnn_512 arcface_512 arcface
    if face_image is not None:
        CVImage(face_image).show()
        CVImage(face_image).save('')

    # # == batch images face detect and align ==
    # fd = FaceDetect5Landmarks(mode='scrfd_500m')
    # img_dir = ''
    # exts = [".jpg", ".png", ".JPG", ".webp", ".jpeg", ".CR2"]
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
    #             CVImage(face_image).save(str(img_save_path))
    #         except (TypeError, cv2.error):
    #             pass

    # # === face detect & tracking ===
    # fd = FaceDetect5Landmarks(mode='scrfd_500m', tracking=True)
    # img_path = 'test_img/test1.jpg'
    # bboxes_mtcnn, kpss_mtcnn = fd.get_bboxes(img_path, max_num=0)
    #
    # img_path = 'test_img/test2.jpg'
    # _, _ = fd.get_bboxes(img_path, max_num=0)
    # face_image, m_ = fd.get_single_face(crop_size=512, mode='default_95')
    # CVImage(face_image, image_format='cv2').show()
    # # fd.draw_face()
