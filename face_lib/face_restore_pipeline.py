# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage
from face_lib.face_restore import GFPGAN
from face_lib.face_detect_and_align import FaceDetect5Landmarks


class FaceRestorePipe:
    def __init__(self, show=False):
        self.show = show

        self.gfp = GFPGAN(model_type='GFPGANv1.4', provider='gpu')
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')

    def face_detect_and_align(self, image_in):
        image_in = CVImage(image_in).bgr
        bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(image_in, min_bbox_size=64)
        # self.fd.draw_face()
        face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=512, mode='mtcnn_512',
                                                                  apply_roi=True)  # mtcnn_512 arcface_512 arcface
        # if face_image is not None:
        if self.show:
            CVImage(face_image_).show(0)
        return face_image_, mat_rev_, roi_box_

    def face_restore(self, src_face_image_):
        return self.gfp.forward(src_face_image_)


if __name__ == '__main__':
    src_img = 'resource/test2.jpg'
    fsp = FaceRestorePipe(show=True)

    face_image, mat_rev, roi_box = fsp.face_detect_and_align(src_img)
    face_restore_out = fsp.face_restore(face_image)

    src_img_in = CVImage(src_img).bgr
    restore_roi = CVImage(None).recover_from_reverse_matrix(face_restore_out,
                                                            src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]],
                                                            mat_rev, img_fg_mask=None)
    src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = restore_roi

    CVImage(src_img_in).show(0)
