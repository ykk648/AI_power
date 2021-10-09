import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np
from ai_utils import down_sample

# https://github.com/neuralchen/SimSwap/blob/main/models/models.py
ARCFACE_MODEL_PATH = 'pretrain_models/arcface_model/arcface_checkpoint.tjm'
# https://github.com/HuangYG123/CurricularFace
CURRICULAR_MODEL_PATH = 'pretrain_models/CurricularFace/CurricularFace.tjm'


class FaceEmbedding:
    def __init__(self, model_type='arc', gpu_ids=None):
        if model_type == 'arc':
            self.facenet = torch.jit.load(ARCFACE_MODEL_PATH)
            self.transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif model_type == 'cur':
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
            face_image = cv2.imread(face_image)
            # face_image = cv2.resize(face_image, (224, 224))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        elif type(face_image) == np.ndarray:
            # print('got np array, assert its cv2 output.')
            pass

        with torch.no_grad():
            face = Image.fromarray(face_image)
            face = self.transformer(face)
            # print(face)
            face = face.unsqueeze(0)
            if self.gpu:
                face = face.cuda()
            # 输入尺寸为(112, 112)  RGB

            face = down_sample(face, size=[112, 112])
            # face = face * 2.0 - 1.0
            # 人脸latent code为512维

            face_latent = self.facenet(face)
            # print(face_latent)
            # norm = torch.norm(face_latent, 2, 1, True)
            # face_latent = torch.div(face_latent, norm)
            face_latent = F.normalize(face_latent, p=2, dim=1)
        return face_latent[0]

