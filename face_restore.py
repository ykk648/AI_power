from face_restore import FaceRestore
from utils.image_io import img_save, img_show
import cv2

if __name__ == '__main__':
    face_img_p = 'test_img/croped_face/512.jpg'
    fa = FaceRestore(use_gpu=False, mode='dfdnet')
    # fa = FaceRestore(mode='gpen')
    face = fa.forward(face_img_p, output_size=256)
    img_show(face)
