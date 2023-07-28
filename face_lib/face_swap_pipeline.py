# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np

from cv2box import CVImage
from face_lib.face_swap import HifiFace
from face_lib.face_swap import InSwapper
from face_lib.face_embedding import FaceEmbedding
from face_lib.face_detect_and_align import FaceDetect5Landmarks


class FaceSwapPipe:
    def __init__(self, mode='hififace', show=False):
        self.show = show
        if mode == 'hififace':
            self.hf = HifiFace(model_name='1195K_bs1', provider='gpu')
            self.fe = FaceEmbedding(model_type='CurricularFace', provider='gpu')
        else:
            self.isw = InSwapper(model_name='inswapper_128', provider='gpu')
            self.fe = FaceEmbedding(model_type='insightface_r50', provider='gpu')
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')

    def face_detect_and_align(self, image_in):
        image_in = CVImage(image_in).bgr
        bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(image_in, min_bbox_size=64)
        # self.fd.draw_face()
        # # print(bboxes_scrfd)
        # face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=512, mode='mtcnn_512', apply_roi=True,
        #                                                           pad_ratio=0.3)
        # face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=256, mode='mtcnn_256', apply_roi=True,
        #                                                           pad_ratio=0.3)
        face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=128, mode='arcface', apply_roi=True,
                                                                  pad_ratio=0.3)
        if self.show:
            CVImage(face_image_).show(0)
        return face_image_, mat_rev_, roi_box_

    def get_face_latent(self, image_in):
        return self.fe.forward(image_in)

    def swap_face_hififace(self, src_img_p, dst_img_p):
        face_image, mat_rev, roi_box = fsp.face_detect_and_align(src_img_p)
        dst_face_latent = fsp.get_face_latent(fsp.face_detect_and_align(dst_img_p)[0]).astype(np.float32)

        face_image = CVImage(face_image).rgb()
        mask, swap_face = self.hf.forward(face_image.astype(np.float32), dst_face_latent[np.newaxis, :])
        swap_face = ((swap_face + 1) / 2).squeeze().transpose(1, 2, 0).astype(np.float32)
        swap_face = CVImage(swap_face).rgb()
        CVImage(swap_face).show(0)

        src_img_in = CVImage(src_img_p).bgr
        swap_roi = CVImage(None).recover_from_reverse_matrix(swap_face,
                                                             src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]],
                                                             mat_rev, img_fg_mask=mask[0][0])
        src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = swap_roi

        CVImage(src_img_in).show(0)
        CVImage(src_img_in).save(src_img.replace('.png', '_swap.png'))
        return

    def swap_face_inswapper(self, src_img_p, dst_img_p):
        face_image, mat_rev, roi_box = fsp.face_detect_and_align(src_img_p)
        dst_face_latent = fsp.get_face_latent(fsp.face_detect_and_align(dst_img_p)[0]).astype(np.float32)

        face_image = CVImage(face_image).rgb()
        swap_face = self.isw.forward(face_image.astype(np.float32), dst_face_latent[np.newaxis, :])[0]
        swap_face = swap_face.squeeze().transpose(1, 2, 0).astype(np.float32)
        swap_face = CVImage(swap_face).rgb()
        CVImage(swap_face).show(0)

        src_img_in = CVImage(src_img_p).bgr
        swap_roi = CVImage(None).recover_from_reverse_matrix(swap_face,
                                                             src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]],
                                                             mat_rev, img_fg_mask=None)
        src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = swap_roi

        CVImage(src_img_in).show(0)
        CVImage(src_img_in).save(src_img.replace('.png', '_swap.png'))
        return


if __name__ == '__main__':
    src_img = 'resources/test1.jpg'
    dst_img = 'resources/test3.jpg'

    # fsp = FaceSwapPipe(mode='hififace', show=True)
    # fsp.swap_face_hififace(src_img, dst_img)

    fsp = FaceSwapPipe(mode='inswapper', show=True)
    fsp.swap_face_inswapper(src_img, dst_img)
