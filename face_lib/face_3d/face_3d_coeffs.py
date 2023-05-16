import torch
import cv2
import numpy as np
from PIL import Image

from .Deep3DFace import define_f3d
from utils.ai_utils import down_sample
from cv2box import CVImage


class Face3dCoeffs:
    def __init__(self, opt=None, gpu_ids=None):
        self.opt = opt
        # 输入224 RGB
        self.f3d_input = 224
        self.f3d = define_f3d(self.opt, gpu_ids=gpu_ids)
        if gpu_ids is not None and len(gpu_ids) > 0:
            self.gpu = True
        else:
            self.gpu = False

    def coeffs_from_image(self, face_image):
        face_image = CVImage(face_image).rgb()

        if type(face_image) in [np.ndarray, str, Image.Image]:
            face_image = np.array(face_image) / 255.0
            face_image = torch.tensor(face_image, dtype=torch.float32)
            face_image = face_image.permute(2, 0, 1).unsqueeze(0)
        elif type(face_image) == torch.Tensor:
            pass

        if self.gpu:
            face_image = face_image.cuda()

        coeffs = self.f3d.compute_coeff((down_sample(face_image, size=self.f3d_input) + 1) / 2)
        return coeffs[0]
