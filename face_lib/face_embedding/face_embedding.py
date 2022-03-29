# -- coding: utf-8 --
# @Time : 2021/11/10
# @Author : ykk648
# @Project : https://github.com/ykk648/AI_power
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np


def down_sample(target_, size):
    return F.interpolate(target_, size=size, mode='bilinear', align_corners=True)


# https://github.com/neuralchen/SimSwap/blob/01a8d6d0a6fd7e7b0052a5832328fba33f2b8414/models/fs_model.py#L63
ARCFACE_MODEL_PATH = 'pretrain_models/face_embedding/ArcFace.tjm'
# https://github.com/HuangYG123/CurricularFace
CURRICULAR_MODEL_PATH = 'pretrain_models/face_embedding/CurricularFace.tjm'
#
W600K_MBF_PATH = 'pretrain_models/face_embedding/w600k_mbf.onnx'


class FaceEmbedding:
    def __init__(self, model_type='arc', gpu_ids=None):
        self.model_type = model_type
        if self.model_type == 'arc':
            self.facenet = torch.jit.load(ARCFACE_MODEL_PATH)
            self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.model_type == 'cur':
            self.facenet = torch.jit.load(CURRICULAR_MODEL_PATH)
            self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        self.facenet.eval()
        if gpu_ids is not None and len(gpu_ids) > 0:
            self.gpu = True
            assert (torch.cuda.is_available())
            self.facenet.to(gpu_ids[0])
            self.facenet = torch.nn.DataParallel(self.facenet, gpu_ids)  # multi-GPUs
        else:
            self.gpu = False

    def latent_from_image(self, face_image):
        if type(face_image) == str:
            face_image = Image.open(face_image).convert('RGB')
            # face_image = cv2.cvtColor(cv2.imread(face_image), cv2.COLOR_BGR2RGB)

        elif type(face_image) == np.ndarray:
            print('Got np array, assert its cv2 output.')
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            face = self.transformer(face_image)
            face = face.unsqueeze(0)
            if self.gpu:
                face = face.cuda()
            if face.shape[2] != face.shape[3] != 112:
                face = down_sample(face, size=[112, 112])
            # input: RGB 0-1  output: 512 embedding
            face_latent = self.facenet(face)
            if self.model_type == 'cur':
                face_latent = F.normalize(face_latent, p=2, dim=1)

        return face_latent[0]


if __name__ == '__main__':
    # # CurricularFace
    # fb_cur = FaceEmbedding(model_type='cur', gpu_ids=[0])
    # latent_cur = fb_cur.latent_from_image('test_img/croped_face/112.png')
    # print(latent_cur.shape)
    # print(latent_cur)

    # ArcFace
    fb_arc = FaceEmbedding(model_type='arc', gpu_ids=[0])
    from cv2box import MyFpsCounter
    with MyFpsCounter() as mfc:
        for i in range(10):
            latent_arc = fb_arc.latent_from_image('test_img/croped_face/112.png')
    print(latent_arc.shape)
    print(latent_arc)
