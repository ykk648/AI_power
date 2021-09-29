from face_align import MTCNN, align_face_ffhq, Face_detect_crop
import cv2
import numpy as np
from ai_utils import img_show, img_save


class FaceAlign:
    def __init__(self, mode='scrfd'):
        self.mode = mode
        self.face_result = None
        if self.mode == 'scrfd':
            # scrfd from insightface
            self.face_detector = Face_detect_crop()
            self.face_detector.prepare(ctx_id=0, det_thresh=0.35, det_size=(640, 640))  # 0.4

        elif self.mode == 'mtcnn':
            # # mtcnn from insightface
            # # https://github.com/taotaonice/FaceShifter
            self.face_detector = MTCNN()
        elif self.mode == 'ffhq':
            # # ffhq align method
            pass

    def forward(self, img_p, output_size=256):
        if self.mode == 'scrfd':
            try:
                img_ = cv2.imread(img_p)
                img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
            except cv2.error:
                if self.verbose:
                    print("img is empty")
                return None

            # detect_results = self.face_detector.get_multi(img_, output_size)
            detect_results = self.face_detector.get_single(img_, output_size)
            if detect_results is None:
                # 未检测到人脸
                if self.verbose:
                    print("src frame no face")
                return None
            else:
                face, mat = detect_results
                face[0] = cv2.cvtColor(face[0], cv2.COLOR_RGB2BGR)
                self.face_result = face[0]
        elif self.mode == 'mtcnn':
            bboxes, faces = self.face_detector.align_multi(img_p, limit=1, min_face_size=30,
                                                           crop_size=(output_size, output_size))
            face_ = np.array(faces[0])[:, :, ::-1]
            self.face_result = face_
        elif self.mode == 'ffhq':
            self.face_result = align_face_ffhq(img_p, output_size=output_size)

        return self.face_result

    def save(self, img_save_p):
        img_save(self.face_result, img_save_p, self.verbose)


if __name__ == '__main__':
    face_img_p = 'test_img/rb.jpeg'
    fa = FaceAlign('scrfd')
    # fa = FaceAlign('mtcnn')
    face = fa.forward(face_img_p, output_size=512)
    img_show(face)
