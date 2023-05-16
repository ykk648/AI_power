# -- coding: utf-8 --
# @Time : 2023/2/22
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import numpy as np

from art_lib.talking_head import TPSMM, KPDetector
from face_lib.face_detect_and_align import FaceDetect5Landmarks

from cv2box import CVVideoLoader, CVImage
from cv2box.cv_gears import Linker, Consumer, CVVideoThread, Queue, CVVideoWriterThread
from tqdm import tqdm


class FaceDetectThread(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')
        self.roi_box = None

    def forward_func(self, something_in):
        frame = something_in[0]
        if self.roi_box is None:
            bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(frame, min_bbox_size=64)
            _, self.roi_box = self.fd.get_single_face(crop_size=(256, 256), only_roi=True,
                                                      pad_ratio=0.25)  # mtcnn_512 arcface_512 arcface

        face_image_ = frame[self.roi_box[1]:self.roi_box[3], self.roi_box[0]:self.roi_box[2]]
        face_image_ = CVImage(face_image_).resize_keep_ratio((256, 256))[0]

        # bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(frame, min_bbox_size=64)
        # face_image_, mat_rev, roi_box = self.fd.get_single_face(crop_size=256, mode='mtcnn_256', apply_roi=True,
        #                                           pad_ratio=0.25)  # mtcnn_512 arcface_512 arcface

        # bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(frame, min_bbox_size=64)
        # face_image_, roi_box = self.fd.get_single_face(crop_size=(256,256), mode='mtcnn_512', only_roi=True,
        #                                           pad_ratio=0.25)  # mtcnn_512 arcface_512 arcface

        # CVImage(face_image_).show(1, 'tst')
        return [face_image_]


class FaceKP(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.kpd = KPDetector(provider='gpu')

    def forward_func(self, something_in):
        drive_image_ = something_in[0]
        drive_image_kp_ = self.kpd.forward(drive_image_)
        # CVImage(out_img_).show(1)
        return [drive_image_kp_, drive_image_]


class FaceDrive(Consumer):
    def __init__(self, queue_list: list, source_img, videoWriter):
        super().__init__(queue_list, fps_counter=True)
        self.source_img = source_img
        self.tpsmm = TPSMM(provider='gpu')
        self.tpsmm.get_kp_source(source_img)

        self.video_writer = videoWriter

    def forward_func(self, something_in):
        drive_image_kp_ = something_in[0]
        out_img_ = self.tpsmm.forward(self.source_img, drive_image_kp_, pass_drive_kp=True)
        # CVImage(out_img_).show(1)
        # self.video_writer.write((out_img_*255).astype(np.uint8))
        self.video_writer.write((something_in[1]).astype(np.uint8))
        return [out_img_]


# class FaceReverseThread(Consumer):
#     def __init__(self, queue_list: list, video_writer):
#         super().__init__(queue_list, fps_counter=True)
#         self.gfp = GFPGAN(model_type='GFPGANv1.4', provider='gpu')
#         self.video_writer = video_writer
#
#     def forward_func(self, something_in):
#         face_restore_out = something_in[0]
#         src_img_in = something_in[1]
#         mat_rev = something_in[3]
#         roi_box = something_in[4]
#         face_mask_ = something_in[5]
#         restore_roi = CVImage(None).recover_from_reverse_matrix(face_restore_out,
#                                                                 src_img_in[roi_box[1]:roi_box[3],
#                                                                 roi_box[0]:roi_box[2]],
#                                                                 mat_rev, img_fg_mask=face_mask_)
#         src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = restore_roi
#
#         # CVImage(src_img_in).show(1)
#         self.video_writer.write(src_img_in)


if __name__ == '__main__':

    video_p = ''
    img_p = ''

    import cv2
    with CVVideoLoader(video_p) as cvvl_dummy:
        video_frame_sum = len(cvvl_dummy)
        videoWriter = cv2.VideoWriter(video_p.replace('.mp4', '_out.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), cvvl_dummy.fps, (256, 256))

    q0 = Queue(2)
    q1 = Queue(2)
    q2 = Queue(2)
    # q3 = Queue(2)

    cvvl = CVVideoThread(video_p, [q0])
    fdt = FaceDetectThread([q0, q1])
    fkp = FaceKP([q1, q2])
    fd = FaceDrive([q2], img_p, videoWriter)

    threads_list = [cvvl, fdt, fkp, fd]

    for thread_ in threads_list:
        thread_.start()

    for thread_ in threads_list:
        thread_.join()

    videoWriter.release()
