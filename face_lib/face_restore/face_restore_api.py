# -- coding: utf-8 --
# @Time : 2021/11/17
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

import cv2
from cv2box import CVImage
from .gpen import GPEN
from .dfdnet import DFDNet
from .gfpgan import GFPGAN


class FaceRestore:
    def __init__(self, use_gpu=True, mode='gpen', verbose=True):
        """
        Args:
            use_gpu:
            mode: gfpgan gfpganv3 gfpganv4 gpen gpen2048 dfdnet RestoreFormer CodeFormer
            verbose:
        """
        self.use_gpu = use_gpu
        self.mode = mode
        self.face_result = None
        self.verbose = verbose
        if self.mode == 'gpen':
            self.fr = GPEN(size=512, use_gpu=self.use_gpu)
        elif self.mode == 'gpen2048':
            self.fr = GPEN(size=2048, use_gpu=self.use_gpu)
        elif self.mode == 'dfdnet':
            self.fr = DFDNet(use_gpu=self.use_gpu)
        elif self.mode == 'gfpgan':
            self.fr = GFPGAN(use_gpu=self.use_gpu, version=2)
        elif self.mode == 'gfpganv3':
            self.fr = GFPGAN(use_gpu=self.use_gpu, version=3)
        elif self.mode == 'gfpganv4':
            self.fr = GFPGAN(use_gpu=self.use_gpu, version=4)
        elif self.mode == 'RestoreFormer':
            self.fr = GFPGAN(use_gpu=self.use_gpu, version='RestoreFormer')
        elif self.mode == 'CodeFormer':
            self.fr = GFPGAN(use_gpu=self.use_gpu, version='CodeFormer')

    def forward(self, img_, output_size=256):
        """
        Args:
            img_: cv2 BGR image or image path
            output_size: output image size
        Returns: cv2 BGR image
        """
        self.face_result = self.fr.forward(img_)
        return cv2.resize(self.face_result, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    def save(self, img_save_p):
        CVImage(self.face_result).save(img_save_p)
        # img_save(self.face_result, img_save_p, self.verbose)


if __name__ == '__main__':
    from cv2box import get_path_by_ext, MyFpsCounter, CVVideoLoader
    from tqdm import tqdm

    # === for image ===
    face_img_p = 'resource/cropped_face/512.jpg'
    fa = FaceRestore(use_gpu=False, mode='gfpganv4')  #
    with MyFpsCounter() as mfc:
        face = fa.forward(face_img_p, output_size=512)
    # CVImage(face, image_format='cv2').save('./gfpgan.jpg')
    CVImage(face, image_format='cv2').show()

    # # === for image dir ===
    # face_img_dir = 'resource/test1'
    # fa = FaceRestore(use_gpu=False, mode='RestoreFormer')
    # for img_p in get_path_by_ext(face_img_dir):
    #     face = fa.forward(str(img_p), output_size=512)
    #     CVImage(face, image_format='cv2').save(str(img_p).replace('.', '_RestoreFormer.'))

    # # === for aligned video ===
    # fa = FaceRestore(use_gpu=True, mode='gfpganv4')
    # video_p = ''
    # video_out_p = ''
    #
    # video_writer = cv2.VideoWriter(video_out_p, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256, 256))
    # with CVVideoLoader(video_p) as cvv:
    #     for _ in tqdm(range(len(cvv))):
    #         success, frame = cvv.get()
    #         frame_out = fa.forward(frame, output_size=256)
    #         # CVImage(frame_out).show()
    #         video_writer.write(frame_out)
