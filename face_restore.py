import torch
import torch.nn.functional as F
import cv2
import numpy as np
from cv2box import flush_print

from ai_utils import img_save, img_show

GPEN_MODEL_PATH = 'pretrain_models/face_restore/gpen/GPEN-512.pth'


class FaceRestore:
    def __init__(self, gpu=True, mode='gpen', verbose=True):
        self.gpu = gpu
        self.mode = mode
        self.face_result = None
        self.verbose = verbose
        if self.mode == 'gpen':
            flush_print('Start init Gpen model !')
            # op init costs time
            from face_restore.gpen.face_gan import FaceGAN
            self.gpen = FaceGAN(model_path=GPEN_MODEL_PATH, size=512,
                                model='GPEN-512', channel_multiplier=2, use_gpu=self.gpu)
            flush_print('Gpen model init done !')

    def _gpen_face_enhance(self, img_):

        if type(img_) == str:
            img_ = cv2.imread(img_)
            return self.gpen.process(img_)
        elif type(img_) == torch.tensor:
            # for tensor
            with torch.no_grad():
                # img origin size
                input_shape = img_.shape[-2:]
                face_tensor = img_ * 2.0 - 1.0
                face_tensor = face_tensor.unsqueeze(0)
                face_tensor = F.interpolate(face_tensor, size=(512, 512))

                enhanced, _ = self.gpen.model(face_tensor)

                enhanced = (enhanced + 1.0) / 2.0
                enhanced = torch.clip(enhanced, min=0.0, max=1.0)
                enhanced = F.interpolate(enhanced, size=input_shape)
                return enhanced[0]
        elif type(img_) == np.array:
            # read from opencv by default
            return self.gpen.process(img_)

    def forward(self, img_, output_size=256):
        if self.mode == 'gpen':
            self.face_result = self._gpen_face_enhance(img_)
        return cv2.resize(self.face_result, (output_size, output_size))

    def save(self, img_save_p):
        img_save(self.face_result, img_save_p, self.verbose)


if __name__ == '__main__':
    face_img_p = 'test_img/rb.png'
    fa = FaceRestore(mode='gpen')
    # fa = FaceAlign('mtcnn')
    face = fa.forward(face_img_p, output_size=112)
    print(face.shape)
    img_show(face)
