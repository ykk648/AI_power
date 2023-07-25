# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np

from cv2box import CVImage
from face_lib.face_swap import HifiFace
from face_lib.face_embedding import FaceEmbedding
from face_lib.face_detect_and_align import FaceDetect5Landmarks


class FaceSwapPipe:
    def __init__(self, show=False):
        self.show = show

        self.hf = HifiFace(model_name='865K_bs1', provider='gpu')
        self.fe = FaceEmbedding(model_type='CurricularFace', provider='gpu')
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')

    def face_detect_and_align(self, image_in):
        image_in = CVImage(image_in).bgr
        bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(image_in, min_bbox_size=64)
        # self.fd.draw_face()
        # # print(bboxes_scrfd)
        # face_image, m_ = self.fd.get_single_face(crop_size=512, mode='mtcnn_512')  # mtcnn_512 arcface_512 arcface
        face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=256, mode='mtcnn_256', apply_roi=True)
        # if face_image is not None:
        if self.show:
            CVImage(face_image_).show(0)
        return face_image_, mat_rev_, roi_box_

    def get_face_latent(self, image_in):
        return self.fe.forward(image_in)

    def swap_face(self, src_face_image_, dst_face_latent_):
        return self.hf.forward(src_face_image_, dst_face_latent_[np.newaxis, :])


if __name__ == '__main__':
    src_img = 'resource/test1.jpg'
    dst_img = 'resource/test3.jpg'
    fsp = FaceSwapPipe(show=True)

    face_image, mat_rev, roi_box = fsp.face_detect_and_align(src_img)
    dst_face_latent = fsp.get_face_latent(fsp.face_detect_and_align(dst_img)[0]).astype(np.float32)
    mask, swap_face = fsp.swap_face(face_image.astype(np.float32), dst_face_latent)

    swap_face = ((swap_face + 1) / 2).squeeze().transpose(1, 2, 0).astype(np.float32)
    CVImage(swap_face).show(0)

    src_img_in = CVImage(src_img).bgr
    swap_roi = CVImage(None).recover_from_reverse_matrix(swap_face, src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]], mat_rev, img_fg_mask=None)
    src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = swap_roi

    CVImage(src_img_in).show(0)
    CVImage(src_img_in).save(src_img.replace('.jpg', '_swap.jpg'))

