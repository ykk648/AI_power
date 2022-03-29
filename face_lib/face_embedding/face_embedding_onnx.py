# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power

from cv2box import CVImage
from cv2box.utils.util import np_norm
from cv2box.utils import CalDistance
from model_lib.onnx_model import ONNXModel

W600K_MBF_PATH = 'pretrain_models/face_embedding/w600k_mbf.onnx'


class FaceEmbeddingONNX:
    def __init__(self, model_type='insightface_mbf'):
        self.facenet = None
        if model_type == 'insightface_mbf':
            self.facenet = ONNXModel(W600K_MBF_PATH)
            self.input_std = self.input_mean = 127.5
            self.input_size = (112, 112)

    def latent_from_image(self, face_image):
        blob = CVImage(face_image).set_blob(self.input_std, self.input_mean, self.input_size).blob_rgb
        return np_norm(self.facenet.forward(blob)[0].ravel())


if __name__ == '__main__':
    # ArcFace MBF
    fe_mbf = FaceEmbeddingONNX(model_type='insightface_mbf')

    latent_mbf_1 = fe_mbf.latent_from_image('test_img/croped_face/112.png')
    latent_mbf_2 = fe_mbf.latent_from_image('test_img/croped_face/112.png')
    # latent_mbf_2 = fe_mbf.latent_from_image('test_img/croped_face/512.jpg')

    print(latent_mbf_1.shape)
    print(latent_mbf_1)
    print(CalDistance().sim(latent_mbf_1, latent_mbf_2))
