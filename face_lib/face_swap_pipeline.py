# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np

from cv2box import CVImage
from face_lib.face_swap import HifiFace, InSwapper, FaceFusionDamo
from face_lib.face_embedding import FaceEmbedding
from face_lib.face_3d import Face3dCoeffs
from face_lib.face_detect_and_align import FaceDetect5Landmarks


class FaceSwapPipe:
    def __init__(self, mode='hififace', show=False):
        self.show = show
        self.mode = mode
        if mode == 'hififace':
            self.hf = HifiFace(model_name='2729K_bs1', provider='gpu')
            self.fe = FaceEmbedding(model_type='CurricularFace', provider='gpu')
        elif mode == 'inswapper':
            self.isw = InSwapper(model_name='inswapper_128', provider='gpu')
            self.fe = FaceEmbedding(model_type='insightface_r50', provider='gpu')
        elif mode == 'facefusion_damo':
            self.ffd = FaceFusionDamo(model_name='face_fusion_damo', provider='gpu')
            self.fe = FaceEmbedding(model_type='CurricularFace', provider='gpu')
            self.d3f = Face3dCoeffs(model_type='facerecon_modelscope', provider='gpu')
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')
        print('model init done!')

    def face_detect_and_align(self, image_in):
        image_in = CVImage(image_in).bgr
        bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(image_in, min_bbox_size=64)
        # self.fd.draw_face()
        if self.mode == 'hififace':
            face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=512, mode='mtcnn_512', apply_roi=True,
                                                                      pad_ratio=0.3)
        elif self.mode == 'inswapper':
            face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=128, mode='arcface', apply_roi=True,
                                                                      pad_ratio=0.3)
        elif self.mode == 'facefusion_damo':
            face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=256, mode='mtcnn_256', apply_roi=True,
                                                                      pad_ratio=0.3)
        if self.show:
            CVImage(face_image_).show(0)
        return face_image_, mat_rev_, roi_box_

    def get_face_latent(self, image_in):
        return self.fe.forward(image_in)

    def swap_face_hififace(self, src_img_p, dst_img_p):
        face_image, mat_rev, roi_box = self.face_detect_and_align(src_img_p)
        dst_face_latent = self.get_face_latent(self.face_detect_and_align(dst_img_p)[0]).astype(np.float32)

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
        return src_img_in

    def swap_face_inswapper(self, src_img_p, dst_img_p):
        face_image, mat_rev, roi_box = self.face_detect_and_align(src_img_p)
        dst_face_latent = self.get_face_latent(self.face_detect_and_align(dst_img_p)[0]).astype(np.float32)

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
        return src_img_in

    def swap_face_fusion_damo(self, src_img_p, dst_img_p):
        # detect
        src_face_image, mat_rev, roi_box = self.face_detect_and_align(src_img_p)
        dst_face_image, _, _ = self.face_detect_and_align(dst_img_p)
        # prepare inputs
        src_face_image = CVImage(src_face_image).bgr
        dst_face_latent = self.get_face_latent(dst_face_image).astype(np.float32)

        src_coeffs = self.d3f.forward(src_face_image)[0]
        dst_coeffs = self.d3f.forward(dst_face_image)[0]
        fuse_coeffs = np.concatenate(((src_coeffs[:, :80] + dst_coeffs[:, :80]) / 2, dst_coeffs[:, 80:]), axis=1)

        def kp_process(coeffs):
            _, _, _, kp = self.d3f.get_3d_params(coeffs)
            kp = kp / 224
            kp[..., 1] = 1 - kp[..., 1]
            return (kp * 2 - 1)[:, :17, :]

        kp_fuse = kp_process(fuse_coeffs)
        kp_dst = kp_process(dst_coeffs)

        # forward
        swap_face = self.ffd.forward(src_face_image, dst_face_latent[np.newaxis, :], kp_fuse, kp_dst)[0]
        swap_face = swap_face.squeeze().transpose(1, 2, 0).astype(np.float32)
        if self.show:
            CVImage(swap_face).show(0)
        # reverse
        src_img_in = CVImage(src_img_p).bgr
        swap_roi = CVImage(None).recover_from_reverse_matrix(swap_face,
                                                             src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]],
                                                             mat_rev, img_fg_mask=None, blur=True)
        if self.show:
            CVImage(swap_roi).show(0)
        src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = swap_roi

        if self.show:
            CVImage(src_img_in).show(0)
        return src_img_in


if __name__ == '__main__':
    src_img = 'resources/test1.jpg'
    dst_img = 'resources/test3.jpg'

    # fsp = FaceSwapPipe(mode='hififace', show=True)
    # fsp.swap_face_hififace(src_img, dst_img)

    fsp = FaceSwapPipe(mode='inswapper', show=True)
    fsp.swap_face_inswapper(src_img, dst_img)

    # fsp = FaceSwapPipe(mode='facefusion_damo', show=True)
    # fsp.swap_face_fusion_damo(src_img, dst_img)
