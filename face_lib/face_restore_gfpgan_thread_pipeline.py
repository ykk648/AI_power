# -- coding: utf-8 --
# @Time : 2022/8/25
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import cv2
from cv2box import CVImage, CVVideoLoader
from cv2box.cv_gears import Linker, Consumer, CVVideoThread, Queue

from face_lib.face_restore import GFPGAN
from face_lib.face_detect_and_align import FaceDetect5Landmarks
from face_lib.face_parsing import FaceParsing


class FaceDetectThread(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.fd = FaceDetect5Landmarks(mode='scrfd_500m')

    def forward_func(self, something_in):
        frame = something_in[0]
        bboxes_scrfd, kpss_scrfd = self.fd.get_bboxes(frame, min_bbox_size=64)
        # self.fd.draw_face()
        face_image_, mat_rev_, roi_box_ = self.fd.get_single_face(crop_size=512, mode='mtcnn_512',
                                                                  apply_roi=True)  # mtcnn_512 arcface_512 arcface
        return [frame, face_image_, mat_rev_, roi_box_]


class FaceRestoreThread(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.gfp = GFPGAN(model_type='GFPGANv1.4', provider='gpu')

    def forward_func(self, something_in):
        src_face_image_ = something_in[1]
        face_restore_out_ = self.gfp.forward(src_face_image_)
        return [face_restore_out_] + something_in


class FaceParseThread(Linker):
    def __init__(self, queue_list: list):
        super().__init__(queue_list, fps_counter=True)
        self.fpt = FaceParsing(model_name='face_parse_onnx', provider='gpu')

    def forward_func(self, something_in):
        face_parse_in_ = something_in[0] * 255
        face_parse_out_ = self.fpt.forward(face_parse_in_)
        face_mask_ = self.fpt.get_face_mask(mask_shape=(512, 512))
        return something_in + [face_mask_]


class FaceReverseThread(Consumer):
    def __init__(self, queue_list: list, video_writer):
        super().__init__(queue_list, fps_counter=True)
        self.gfp = GFPGAN(model_type='GFPGANv1.4', provider='gpu')
        self.video_writer = video_writer

    def forward_func(self, something_in):
        face_restore_out = something_in[0]
        src_img_in = something_in[1]
        mat_rev = something_in[3]
        roi_box = something_in[4]
        face_mask_ = something_in[5]
        restore_roi = CVImage(None).recover_from_reverse_matrix(face_restore_out,
                                                                src_img_in[roi_box[1]:roi_box[3],
                                                                roi_box[0]:roi_box[2]],
                                                                mat_rev, img_fg_mask=face_mask_)
        src_img_in[roi_box[1]:roi_box[3], roi_box[0]:roi_box[2]] = restore_roi

        # CVImage(src_img_in).show(1)
        self.video_writer.write(src_img_in)


if __name__ == '__main__':
    video_in_p = ''
    video_out_p = ''

    with CVVideoLoader(video_in_p) as cvvl_dummy:
        videoWriter = cv2.VideoWriter(video_out_p, cv2.VideoWriter_fourcc(*'mp4v'), cvvl_dummy.fps, cvvl_dummy.size)

    q0 = Queue(2)
    q1 = Queue(2)
    q2 = Queue(2)
    q3 = Queue(2)

    cvvl = CVVideoThread(video_in_p, [q0])
    fdt = FaceDetectThread([q0, q1])
    frt = FaceRestoreThread([q1, q2])
    fpt = FaceParseThread([q2, q3])
    fret = FaceReverseThread([q3], videoWriter)

    threads_list = [cvvl, fdt, frt, fpt, fret]

    for thread_ in threads_list:
        thread_.start()

    for thread_ in threads_list:
        thread_.join()

    videoWriter.release()
